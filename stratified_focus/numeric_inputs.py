"""Consistent decimal and percentage entry for the desktop interface."""
from PyQt6.QtCore import QLocale
from PyQt6.QtWidgets import QDoubleSpinBox


def parse_decimal(text):
    return float(text.strip().replace(',', '.'))


class DecimalSpinBox(QDoubleSpinBox):
    """Accept either decimal separator; display values consistently with a dot."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLocale(QLocale.c())
        self.setGroupSeparatorShown(False)

    def validate(self, text, pos):
        return super().validate(text.replace(',', '.'), pos)

    def valueFromText(self, text):
        return super().valueFromText(text.replace(',', '.'))


class PercentageSpinBox(DecimalSpinBox):
    """Display percentages while exposing fractions to rendering and export code."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSuffix(' [%]')

    def value(self):
        return super().value()/100.

    def setValue(self, fraction):
        super().setValue(100*fraction)

    def setRange(self, low, high):
        super().setRange(100*low,100*high)

    def setSingleStep(self, fraction):
        super().setSingleStep(100*fraction)
