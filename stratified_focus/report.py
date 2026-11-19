"""Concise, printable simulation report for an exported session."""
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import numpy as np
from matplotlib import get_data_path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from .analysis import intensity


FIGURE_NAMES = ('Multilayer Fresnel coefficients',
                'Pupil field components and phases',
                'XY focal intensity',
                'XY field components and phases',
                'XZ and YZ focal intensity',
                'Polarization ellipse axes',
                'XYZ volume intensity')


def _fmt(value):
    if value is None:
        return 'Infinite'  # thickness of a semi-infinite medium
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, (int, float)):
        return f'{value:.8g}'
    if isinstance(value, (tuple, list)):
        return ', '.join(_fmt(v) for v in value)
    return str(value)


def build_report(path, result, metadata, figure_paths):
    """Write a PDF with provenance, every stored input, metrics and figure overviews."""
    fonts = Path(get_data_path()) / 'fonts' / 'ttf'
    pdfmetrics.registerFont(TTFont('FocusSans', str(fonts / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('FocusSansBold', str(fonts / 'DejaVuSans-Bold.ttf')))
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        if hasattr(style, 'fontName'):
            style.fontName = 'FocusSansBold' if 'Bold' in style.fontName else 'FocusSans'
    styles.add(ParagraphStyle(name='ReportTitle', parent=styles['Title'],
                              fontName='FocusSansBold', fontSize=20, leading=24,
                              textColor=colors.HexColor('#173e61'), spaceAfter=13))
    styles.add(ParagraphStyle(name='ReportSubtitle', parent=styles['Normal'],
                              fontSize=9, leading=13, textColor=colors.HexColor('#42566b'),
                              spaceAfter=12))
    styles.add(ParagraphStyle(name='ReportFigure', parent=styles['Heading1'],
                              alignment=TA_CENTER, spaceAfter=14))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18*mm,
                            rightMargin=18*mm, topMargin=18*mm, bottomMargin=17*mm)
    story = [Paragraph('MultiLayerFocus | Simulation Report', styles['ReportTitle'])]
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
    story.append(Paragraph(f'Exported: {escape(stamp)}<br/>Simulation saved: '
                           f'{escape(str(metadata.get("created_at", "Unknown")))}<br/>'
                           f'{escape(str(metadata.get("software", "Unknown")))}',
                           styles['ReportSubtitle']))
    story.append(Paragraph('All distances are measured in micrometers [µm]; phase is measured in radians [rad]. '
                           'Field intensity is numerically normalized, not calibrated power.',
                           styles['Normal']))
    story.append(Spacer(1, 7*mm))

    def section(title, rows):
        story.append(Paragraph(title, styles['Heading2']))
        wrapped = [[Paragraph(escape(str(k)), styles['Normal']),
                    Paragraph(escape(str(v)), styles['Normal'])] for k, v in rows]
        table = Table(wrapped, colWidths=(56*mm, 118*mm), hAlign='LEFT')
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                                   ('ROWBACKGROUNDS',(0,0),(-1,-1),
                                    [colors.white, colors.HexColor('#f1f5f9')]),
                                   ('BOTTOMPADDING',(0,0),(-1,-1),5),
                                   ('TOPPADDING',(0,0),(-1,-1),5),
                                   ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#b8c8d6'))]))
        story.append(table)
        story.append(Spacer(1, 5*mm))

    field = result.xy
    power = np.abs(field)**2
    parts = np.trapezoid(np.trapezoid(power,result.x,axis=2),result.y,axis=1)
    total = parts.sum()
    section('Computed results', [
        ('Compute time (s)', _fmt(result.elapsed_seconds) if result.elapsed_seconds is not None else 'Unknown'),
        ('XY normalisation factor', _fmt(result.normalization)),
        ('XY integrated intensity', _fmt(total)),
        ('Peak XY intensity', _fmt(intensity(field).max())),
        ('XY component contributions', ', '.join(f'E{axis}: {100*part/total:.2f}%'
                                                  for axis, part in zip('xyz',parts))),
        ('Axial planes computed', _fmt(result.xz is not None)),
        ('XYZ volume computed', _fmt(result.volume is not None)),
        ('Volume computation time (s)', _fmt(result.volume_seconds) if result.volume_seconds is not None else 'Not computed'),
    ])
    config = metadata.get('config')
    if config is None:
        story.append(Paragraph('Full input parameters were not recorded in this earlier NPZ. '
                               'The measured arrays remain available for plotting.', styles['Normal']))
        section('Available saved data', [
            ('Incident media indices', _fmt(result.config.stack.indices)),
            ('Layer thicknesses [µm]', _fmt(result.config.stack.thicknesses)),
            ('Vacuum wavelength [µm]', _fmt(result.config.illumination.wavelength)),
            ('NA', _fmt(result.config.objective.na)),
            ('XY array shape', _fmt(result.xy.shape)),
            ('Pupil array shape', _fmt(result.pupil_ex.shape)),
        ])
    else:
        for key, title in (('objective','Objective'), ('illumination','Illumination'),
                           ('sampling','Spatial and angular sampling')):
            section(title, [(k.replace('_',' ').capitalize(), _fmt(v))
                            for k,v in config[key].items()])
        stack = config['stack']
        indices = [f'{v[0]:g}{v[1]:+g}i' for v in stack['indices']]
        section('Multilayer (incident to exit)', [
            ('Refractive indices', ', '.join(indices)),
            ('Thicknesses [µm]', _fmt(stack['thicknesses'])),
            ('Last interface z [µm]', _fmt(stack['z_out'])),
            ('First interface z [µm]', _fmt(result.config.stack.z_in)),
        ])
        section('Pupil field provenance', [
            ('External source filenames', ', '.join(metadata.get('source_files',[])) or 'Generated pupil'),
            ('Snapshot format version', _fmt(metadata['schema_version'])),
        ])

    for j, figure in enumerate(figure_paths):
        story.append(PageBreak())
        story.append(Paragraph(f'Figure {j+1}. {FIGURE_NAMES[j]}', styles['ReportFigure']))
        if figure is None:
            story.append(Paragraph('XYZ volume was not computed for this simulation.' if j == 6 else
                                   'Axial planes were not computed for this simulation.', styles['Normal']))
            continue
        # File-like PNGs keep the PDF overviews in memory. Windows can retain an
        # open handle on disk images until ReportLab finishes laying out pages.
        image = Image(figure if hasattr(figure, 'read') else str(figure))
        scale = min((174*mm)/image.imageWidth, (240*mm)/image.imageHeight)
        image.drawWidth = image.imageWidth*scale
        image.drawHeight = image.imageHeight*scale
        story.append(image)

    def footer(canvas, pdf):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#cbd5df'))
        canvas.line(18*mm, 14*mm, 192*mm, 14*mm)
        canvas.setFont('FocusSans', 8)
        canvas.setFillColor(colors.HexColor('#536b80'))
        canvas.drawString(18*mm, 10*mm, 'MultiLayerFocus | Simulation Report')
        canvas.drawRightString(192*mm, 10*mm, f'Page {pdf.page}')
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
