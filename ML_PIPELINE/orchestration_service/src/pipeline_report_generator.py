#!/usr/bin/env python3
"""
Generate a PDF report for ML pipeline results
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PipelineReportGenerator:
    """Generate PDF reports from pipeline results"""
    
    def __init__(self, output_path="pipeline_report.pdf"):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # Center
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=12,
            spaceBefore=12
        ))
    
    def generate(self, pipeline_results):
        """
        Generate PDF report from pipeline results
        
        Args:
            pipeline_results: Dict with keys:
                - pipeline_id
                - timestamp
                - config (osd_ids, target_column, algorithm, etc.)
                - training_results (metrics, model_id, etc.)
                - feature_importance (method -> [features])
                - ensemble_results (if applicable)
        """
        try:
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=letter,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            story = []
            
            # Title page
            story.extend(self._create_title_page(pipeline_results))
            story.append(PageBreak())
            
            # Configuration
            story.extend(self._create_config_section(pipeline_results.get('config', {})))
            story.append(Spacer(1, 0.2*inch))
            
            # Training results
            story.extend(self._create_training_section(pipeline_results.get('training_results', {})))
            story.append(Spacer(1, 0.2*inch))
            
            # Feature importance
            fi_data = pipeline_results.get('feature_importance', {})
            if fi_data:
                story.append(PageBreak())
                story.extend(self._create_feature_importance_section(fi_data))
                story.append(Spacer(1, 0.2*inch))
            
            # Ensemble results
            ensemble_data = pipeline_results.get('ensemble_results', {})
            if ensemble_data:
                story.append(PageBreak())
                story.extend(self._create_ensemble_section(ensemble_data))
            
            # Build PDF
            doc.build(story)
            logger.info(f"✓ PDF report generated: {self.output_path}")
            return self.output_path
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}", exc_info=True)
            raise
    
    def _create_title_page(self, results):
        """Create title page"""
        story = []
        
        story.append(Spacer(1, 1*inch))
        story.append(Paragraph("ML Pipeline Experiment Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 0.3*inch))
        
        # Metadata
        timestamp = results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        pipeline_id = results.get('pipeline_id', 'Unknown')
        
        metadata_style = self.styles['Normal']
        story.append(Paragraph(f"<b>Pipeline ID:</b> {pipeline_id}", metadata_style))
        story.append(Paragraph(f"<b>Generated:</b> {timestamp}", metadata_style))
        story.append(Spacer(1, 0.2*inch))
        
        config = results.get('config', {})
        story.append(Paragraph(f"<b>Target Column:</b> {config.get('target_column', 'N/A')}", metadata_style))
        story.append(Paragraph(f"<b>Algorithm:</b> {config.get('algorithm', 'N/A')}", metadata_style))
        story.append(Paragraph(f"<b>OSD IDs:</b> {', '.join(config.get('osd_ids', []))}", metadata_style))
        
        return story
    
    def _create_config_section(self, config):
        """Create configuration section"""
        story = []
        
        story.append(Paragraph("Configuration", self.styles['SectionHeading']))
        
        config_data = [
            ['Parameter', 'Value'],
            ['OSD IDs', ', '.join(config.get('osd_ids', []))],
            ['Target Column', config.get('target_column', 'N/A')],
            ['Algorithm', config.get('algorithm', 'N/A')],
            ['Test Size', f"{config.get('test_size', 0.2):.1%}"],
            ['Min Features', str(config.get('min_features', 1000))],
            ['Transformations', ', '.join(config.get('transformations', [])) or 'None'],
        ]
        
        table = Table(config_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        
        story.append(table)
        return story
    
    def _create_training_section(self, results):
        """Create training results section"""
        story = []
        
        story.append(Paragraph("Training Results", self.styles['SectionHeading']))
        
        if not results:
            story.append(Paragraph("No training results available", self.styles['Normal']))
            return story
        
        # Model metrics
        metrics = results.get('metrics', {})
        if metrics:
            story.append(Paragraph("<b>Model Metrics:</b>", self.styles['Normal']))
            
            metrics_data = [['Metric', 'Value']]
            for metric_name, value in metrics.items():
                metrics_data.append([metric_name, f'{value:.4f}' if isinstance(value, float) else str(value)])
            
            table = Table(metrics_data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
        
        # Other information
        model_id = results.get('model_id', 'N/A')
        n_samples = results.get('n_samples', 'N/A')
        n_features = results.get('n_features', 'N/A')
        
        story.append(Paragraph(f"<b>Model ID:</b> {model_id}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Training Samples:</b> {n_samples}", self.styles['Normal']))
        story.append(Paragraph(f"<b>Features Used:</b> {n_features}", self.styles['Normal']))
        
        return story
    
    def _create_feature_importance_section(self, fi_data):
        """Create feature importance section"""
        story = []
        
        story.append(Paragraph("Feature Importance Analysis", self.styles['SectionHeading']))
        
        # Show top features for each method
        for method, features in fi_data.items():
            story.append(Paragraph(f"<b>{method.title()}</b>", self.styles['Normal']))
            
            if not features:
                story.append(Paragraph("No features found", self.styles['Normal']))
                continue
            
            # Top 10 features
            top_features = features[:10]
            fi_table_data = [['Rank', 'Feature', 'Importance']]
            
            for feature in top_features:
                rank = feature.get('rank', '?')
                name = feature.get('feature_name', '?')
                importance = feature.get('importance', 0)
                fi_table_data.append([str(rank), name, f'{importance:.4f}'])
            
            table = Table(fi_table_data, colWidths=[0.5*inch, 3*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.15*inch))
        
        return story
    
    def _create_ensemble_section(self, ensemble_data):
        """Create ensemble results section"""
        story = []
        
        story.append(Paragraph("Ensemble Results", self.styles['SectionHeading']))
        
        # Ensemble metrics
        metrics = ensemble_data.get('metrics', {})
        if metrics:
            story.append(Paragraph("<b>Ensemble Metrics:</b>", self.styles['Normal']))
            
            metrics_data = [['Metric', 'Value']]
            for metric_name, value in metrics.items():
                metrics_data.append([metric_name, f'{value:.4f}' if isinstance(value, float) else str(value)])
            
            table = Table(metrics_data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 0.2*inch))
        
        return story


def generate_pdf_report(pipeline_results, output_path="pipeline_report.pdf"):
    """
    Convenience function to generate a PDF report
    
    Args:
        pipeline_results: Dict with pipeline execution results
        output_path: Where to save the PDF
    
    Returns:
        Path to the generated PDF
    """
    generator = PipelineReportGenerator(output_path)
    return generator.generate(pipeline_results)


if __name__ == "__main__":
    # Example usage
    example_results = {
        'pipeline_id': 'test-pipeline-123',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'osd_ids': ['104', '48'],
            'target_column': 'Factor Value[Spaceflight]',
            'algorithm': 'random_forest',
            'test_size': 0.2,
            'min_features': 1000,
            'transformations': ['standardize', 'log']
        },
        'training_results': {
            'model_id': 'model-abc123',
            'n_samples': 24,
            'n_features': 101,
            'metrics': {
                'accuracy': 0.8521,
                'precision': 0.8333,
                'recall': 0.8571,
                'f1_score': 0.8451
            }
        },
        'feature_importance': {
            'random_forest': [
                {'rank': 1, 'feature_name': 'ENSMUSG00000000001', 'importance': 0.0845},
                {'rank': 2, 'feature_name': 'ENSMUSG00000000002', 'importance': 0.0632},
                {'rank': 3, 'feature_name': 'ENSMUSG00000000003', 'importance': 0.0521},
            ]
        }
    }
    
    pdf_path = generate_pdf_report(example_results, "/tmp/test_report.pdf")
    print(f"Report generated at: {pdf_path}")
