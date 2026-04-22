# test_download_dataset.py
"""
Integration test for the DownloadDataset RPC.

Usage:
    python test_download_dataset.py                  # uses default OSD-379
    python test_download_dataset.py 622              # test a specific OSD ID
    python test_download_dataset.py 379 --patterns Unnormalized RSEM
"""

import sys
import argparse
import grpc
from data_service.generated import data_service_pb2, data_service_pb2_grpc

DATA_SERVICE_HOST = "localhost:50051"

# ── helpers ──────────────────────────────────────────────────────────────────

def get_stub():
    channel = grpc.insecure_channel(DATA_SERVICE_HOST)
    return data_service_pb2_grpc.DataServiceStub(channel)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_result(result):
    print(f"  is_valid : {result.is_valid}")
    if result.errors:
        print(f"  errors   : {list(result.errors)}")
    if result.warnings:
        print(f"  warnings : {list(result.warnings)}")
    if result.info and result.info.dataset_id:
        info = result.info
        print(f"  dataset_id  : {info.dataset_id}")
        print(f"  rows        : {info.num_rows}")
        print(f"  columns     : {info.num_columns}")
        print(f"  size_bytes  : {info.size_bytes}")
        if info.columns:
            print(f"  first 5 cols: {[c.name for c in info.columns[:5]]}")


# ── test cases ────────────────────────────────────────────────────────────────

def test_basic_download(stub, osd_id: str, patterns: list):
    """Happy path: download a real NASA OSDR dataset."""
    print_section(f"TEST 1: Basic download  OSD-{osd_id}  patterns={patterns}")

    request = data_service_pb2.DownloadRequest(
        osd_id=osd_id,
        patterns=patterns,
    )
    result = stub.DownloadDataset(request)
    print_result(result)

    assert result.is_valid, f"Expected valid dataset, got errors: {list(result.errors)}"
    assert result.info.dataset_id, "Expected a dataset_id in response"
    assert result.info.num_rows > 0, "Expected at least one row"
    assert result.info.num_columns > 0, "Expected at least one column"
    print("  ✓ PASSED")
    return result.info.dataset_id   # return for downstream tests


def test_custom_dataset_id(stub, osd_id: str, patterns: list):
    """Caller-supplied dataset_id should be echoed back."""
    print_section("TEST 2: Custom dataset_id is preserved")

    custom_id = "my-custom-id-abc123"
    request = data_service_pb2.DownloadRequest(
        osd_id=osd_id,
        patterns=patterns,
        dataset_id=custom_id,
    )
    result = stub.DownloadDataset(request)
    print_result(result)

    assert result.is_valid, f"Download failed: {list(result.errors)}"
    assert result.info.dataset_id == custom_id, (
        f"Expected dataset_id '{custom_id}', got '{result.info.dataset_id}'"
    )
    print("  ✓ PASSED")
    return result.info.dataset_id


def test_dataset_available_for_streaming(stub, dataset_id: str):
    """Dataset stored by DownloadDataset should be streamable."""
    print_section(f"TEST 3: Streamed dataset  id={dataset_id}")

    request = data_service_pb2.StreamRequest(
        dataset_id=dataset_id,
        chunk_size=500,
    )
    chunks = list(stub.StreamDataset(request))

    assert len(chunks) > 0, "Expected at least one chunk"
    assert chunks[-1].is_final, "Last chunk should have is_final=True"

    total_rows = 0
    for chunk in chunks:
        lines = chunk.data.decode("utf-8").strip().split("\n")
        # subtract 1 for the header row present in every chunk
        total_rows += max(len(lines) - 1, 0)

    print(f"  chunks received : {len(chunks)}")
    print(f"  total data rows : {total_rows}")
    print("  ✓ PASSED")


def test_invalid_osd_id(stub):
    """Non-existent OSD ID should return is_valid=False, not crash."""
    print_section("TEST 4: Invalid OSD ID")

    request = data_service_pb2.DownloadRequest(
        osd_id="999999",
        patterns=["Unnormalized", "RSEM"],
    )
    result = stub.DownloadDataset(request)
    print_result(result)

    assert not result.is_valid, "Expected is_valid=False for a bad OSD ID"
    assert result.errors, "Expected at least one error message"
    print("  ✓ PASSED")


def test_no_pattern_match(stub, osd_id: str):
    """Patterns that match nothing should return is_valid=False."""
    print_section("TEST 5: Patterns with no matches")

    request = data_service_pb2.DownloadRequest(
        osd_id=osd_id,
        patterns=["ZZZNOMATCH123"],
    )
    result = stub.DownloadDataset(request)
    print_result(result)

    assert not result.is_valid, "Expected is_valid=False when no files match"
    assert result.errors, "Expected an error message"
    print("  ✓ PASSED")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test DownloadDataset RPC")
    parser.add_argument("osd_id", nargs="?", default="104",
                        help="NASA OSDR dataset number (default: 104)")
    parser.add_argument("--patterns", nargs="+", default=["Unnormalized", "RSEM"],
                        help="File patterns to match (default: Unnormalized RSEM)")
    parser.add_argument("--host", default=DATA_SERVICE_HOST,
                        help=f"gRPC host:port (default: {DATA_SERVICE_HOST})")
    args = parser.parse_args()

    stub = get_stub()
    passed = 0
    failed = 0

    tests = [
        lambda: test_basic_download(stub, args.osd_id, args.patterns),
        lambda: test_custom_dataset_id(stub, args.osd_id, args.patterns),
        lambda: test_invalid_osd_id(stub),
        lambda: test_no_pattern_match(stub, args.osd_id),
    ]

    # Run tests 1 & 2 first and capture dataset_id for streaming test
    dataset_id = None
    for test_fn in tests[:2]:
        try:
            result = test_fn()
            if dataset_id is None:
                dataset_id = result   # grab id from test 1
            passed += 1
        except (AssertionError, grpc.RpcError) as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    # Streaming test depends on a valid dataset_id from test 1
    if dataset_id:
        try:
            test_dataset_available_for_streaming(stub, dataset_id)
            passed += 1
        except (AssertionError, grpc.RpcError) as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    else:
        print("\n  SKIPPED streaming test (no dataset_id from download)")
        failed += 1

    # Error-case tests
    for test_fn in tests[2:]:
        try:
            test_fn()
            passed += 1
        except (AssertionError, grpc.RpcError) as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print_section(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
