"""Genereer het officiële Deel 2 PDF-rapport en bijbehorende visualisaties.

Draaien:
python -m src.generate_report
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import confusion_matrix, recall_score

from src.config import laad_config
from src.data import laad_data
from src.train import bouw_pijplijn, splits

PROJECT = Path(__file__).resolve().parent.parent


def genereer_grafiek(df, cfg, plot_pad: Path):
    X_train, X_test, y_train, y_test = splits(df, cfg)
    threshold = cfg.get("threshold", 0.15)

    lr = bouw_pijplijn(cfg, "logistic_regression").fit(X_train, y_train)
    rf = bouw_pijplijn(cfg, "random_forest").fit(X_train, y_train)

    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= threshold).astype(int)
    cm_lr = confusion_matrix(y_test, y_pred_lr)

    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf >= threshold).astype(int)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=300)

    rec_lr = recall_score(y_test, y_pred_lr)
    rec_rf = recall_score(y_test, y_pred_rf)

    titles = [
        f"Baseline: Logistic Regression\nRecall: {rec_lr:.1%} | FP: {cm_lr[0, 1]}",
        f"Gekozen: Random Forest\nRecall: {rec_rf:.1%} | FP: {cm_rf[0, 1]}",
    ]

    for ax, cm, title, color_map in zip(
        axes,
        [cm_lr, cm_rf],
        titles,
        ["Oranges", "Blues"],
        strict=True,
    ):
        im = ax.imshow(cm, interpolation="nearest", cmap=color_map)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(["Normaal", "Defect"], fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(["Normaal", "Defect"], fontsize=8)
        ax.set_xlabel("Voorspeld (Drempel 0.15)", fontsize=8)
        ax.set_ylabel("Werkelijk", fontsize=8)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(plot_pad, dpi=300, bbox_inches="tight")
    plt.close()


def genereer_pdf(pdf_pad: Path, plot_pad: Path):
    doc = SimpleDocTemplate(
        str(pdf_pad),
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()

    # Kleurenpalet
    primary = colors.HexColor("#1A365D")
    text_dark = colors.HexColor("#2D3748")
    accent_green = colors.HexColor("#22543D")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=primary,
        spaceAfter=4,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=primary,
        spaceBefore=7,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=text_dark,
        spaceAfter=4,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=body_style,
        fontSize=8.2,
        leading=11,
        textColor=accent_green,
        backColor=colors.HexColor("#F0FFF4"),
        borderColor=colors.HexColor("#9AE6B4"),
        borderWidth=0.8,
        borderPadding=5,
        spaceAfter=5,
    )

    story = []

    # ==================== PAGINA 1 ====================
    story.append(Paragraph("Cloud for AI: Deel 2 — ML-Oplossing & Modelvergelijking", title_style))
    story.append(
        Paragraph(
            "<b>Auteur:</b> Joussef Boulbane &nbsp;|&nbsp; "
            "<b>Dataset:</b> AI4I 2020 Predictive Maintenance (10.000 rijen) &nbsp;|&nbsp; "
            "<b>Repository:</b> <font color='#2B6CB0'>github.com/JoussefB/cloud-for-ai</font>",
            meta_style,
        )
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=primary,
            spaceBefore=0,
            spaceAfter=6,
        )
    )

    # Sectie 1
    p1 = (
        "In navolging van het bedrijfsbezoek aan BMSVision richt dit project zich op "
        "<b>predictive maintenance</b> voor frees- en productiemachines. Het doel is "
        "machinefalen tijdig te detecteren om ongeplande productiestilstand te voorkomen. "
        "In <i>Deel 1 (Scoping)</i> is vastgelegd dat <b>Recall</b> de leidende ML-metriek is "
        "wegens de extreme asymmetrie in de foutkosten: een gemist defect (Vals Negatief) kost "
        "circa <b>€10.000</b> (gebroken gereedschap, stilstand, gevolgschade), terwijl een "
        "onnodige controle (Vals Positief) slechts circa <b>€150</b> kost (15 min inspectie). "
        "Om deze reden hanteren we een verlaagde classificatiedrempelwaarde van <b>0.15</b>."
    )
    story.append(Paragraph("1. Inleiding & Evaluatiekader", h1_style))
    story.append(Paragraph(p1, body_style))

    # Sectie 2
    p2 = (
        "De gebruikte dataset bevat 10.000 rijen en 14 kolommen. Bij de data-exploratie zijn "
        "de volgende observaties gedaan: <br/>"
        "• <b>Sterke klasse-onbalans:</b> Slechts 339 van de 10.000 observaties (<b>3,39%</b>) "
        "vertonen feitelijk falen (`Machine failure = 1`). Een naïeve classifier die altijd 'geen defect' "
        "voorspelt haalt 96,6% accuracy maar 0% recall. Oplossing: `class_weight='balanced'` in de "
        "modellen gecombineerd met de 0.15-threshold.<br/>"
        "• <b>Geen ontbrekende waarden:</b> De dataset vertoont 0 null-waarden, maar de bereiken "
        "vereisen strikte grenzen (luchttemp. 295–305 K, toerental 1168–2886 rpm, slijtage 0–253 min).<br/>"
        "• <b>Geautomatiseerde data-integriteit:</b> Via een <code>pandera.DataFrameSchema</code> "
        "(<code>src/schema.py</code>) wordt elke batch gevalideerd op types, unieke ID's, categorische "
        "restricties (L/M/H) en fysiek geldige sensorbereiken. Ongeldige batches falen direct."
    )
    story.append(Paragraph("2. Data-exploratie, Datakwaliteit & Pandera-validatie", h1_style))
    story.append(Paragraph(p2, body_style))

    # Sectie 3
    p3 = (
        "We hebben conform de opdracht twee modellen getraind en geëvalueerd op een "
        "gestratificeerde testset (20% = 2.000 rijen, waarvan 68 defecten):<br/>"
        "1. <b>Baseline Model:</b> <i>Logistic Regression</i> met L2-regularisatie en "
        "<code>class_weight='balanced'</code>. Doelbewust eenvoudig, transparant en snel te fitten.<br/>"
        "2. <b>Challenger Model:</b> <i>Random Forest Classifier</i> (200 beslissingsbomen, balanced "
        "subsample weighting). Vangt niet-lineaire interacties tussen sensoren (koppel × toerental)."
    )
    story.append(Paragraph("3. Modelvergelijking & Experimentele Resultaten", h1_style))
    story.append(Paragraph(p3, body_style))

    # Resultatentabel
    tabel_data = [
        [
            "Model",
            "Recall (drempel 0.15)",
            "Precision",
            "F1-Score",
            "Accuracy",
            "Gemiste Defecten (FN)",
            "Valse Alarmen (FP)",
        ],
        [
            "Logistic Regression (Baseline)",
            "95.6% (65/68)",
            "6.2%",
            "0.117",
            "50.8%",
            "3",
            "979",
        ],
        [
            "Random Forest (Gekozen)",
            "91.2% (62/68)",
            "34.8%",
            "0.504",
            "94.0%",
            "6",
            "116",
        ],
    ]
    t = Table(tabel_data, colWidths=[120, 85, 45, 45, 45, 90, 80])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), primary),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7.5),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#EBF8FF")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 4))

    callout_txt = (
        "<b>Besparingseffect:</b> Uit de 68 teststoringen vangt de baseline er 65 "
        "(besparing: 65 × €10.000 = €650.000; kosten FP: 979 × €150 = €146.850; "
        "<i>Netto winst: €503.150</i>). "
        "Random Forest vangt er 62 (besparing: 62 × €10.000 = €620.000; kosten FP: 116 × €150 = €17.400; "
        "<i>Netto winst: €602.600</i>). "
        "Random Forest levert direct <b>€99.450 méér netto economisch voordeel</b> op over 2.000 cycli."
    )
    story.append(Paragraph(callout_txt, callout_style))

    # ==================== PAGINA 2 ====================
    story.append(PageBreak())

    p4 = (
        "Hoewel <i>Logistic Regression</i> op papier een fractioneel hogere ruwe recall noteert "
        "(95,6% vs 91,2%), kiezen we beslist voor <b>Random Forest</b> voor deployment. "
        "De argumenten hiervoor zijn doorslaggevend:<br/>"
        "• <b>Beheersen van Alert Fatigue:</b> De baseline slaat in 979 van de 1.932 gezonde gevallen "
        "vals alarm (precisie slechts 6,2%). Technici worden bedolven onder meldingen en verliezen "
        "vertrouwen in het systeem. Random Forest reduceert het aantal valse meldingen met meer dan "
        "<b>88%</b> (van 979 naar 116), waardoor technici gericht en serieus reageren.<br/>"
        "• <b>Fysieke Interacties:</b> Random Forest slaagt erin samengestelde faalfenomenen "
        "(zoals oververhitting bij lage toerentallen of overbelasting door hoog koppel en gereedschapsslijtage) "
        "niet-lineair te scheiden zonder manuele feature engineering.<br/>"
        "• <b>Softwaremetriek (Latency):</b> De gemiddelde inferentietijd van Random Forest bedraagt "
        "<b>0,85 ms per machine</b>. Dit blijft ver onder de software-eis van < 1 seconde."
    )
    story.append(Paragraph("4. Verantwoording Modelkeuze: Waarom Random Forest?", h1_style))
    story.append(Paragraph(p4, body_style))
    story.append(Spacer(1, 2))
    story.append(Image(str(plot_pad), width=510, height=210))
    story.append(Spacer(1, 4))

    # Sectie 5: Data Leakage
    p5_a = (
        "<b>Waar zat het risico in de data?</b><br/>"
        "In de AI4I 2020 dataset schuilt een levensgroot risico op data leakage in de vijf "
        "diagnostische sub-storingskolommen: <code>TWF</code> (Tool Wear Failure), <code>HDF</code> "
        "(Heat Dissipation Failure), <code>PWF</code> (Power Failure), <code>OSF</code> (Overstrain Failure) "
        "en <code>RNF</code> (Random Failure). Deze kolommen zijn deterministische deelcomponenten "
        "van de doelvariabele <code>Machine failure</code>. Zou men deze kolommen in de featurematrix "
        "opnemen, dan leert het algoritme een triviale identiteitsfunctie (100% target leakage). "
        "In een operationele fabriek zijn deze exacte diagnoses pas <i>achteraf</i> na demontage van de machine "
        "bekend, nooit vooraf tijdens continue sensormonitoring. We hebben deze kolommen, evenals niet-generieke "
        "identifier-kolommen (<code>UDI</code> en <code>Product ID</code>), <b>strikt geëxcludeerd</b> "
        "uit de modelconfiguratie (<code>configs/default.yaml</code>)."
    )
    p5_b = (
        "<b>Hoe weten we dat de split eerlijk is?</b><br/>"
        "De datasplit is strikt hermetisch geïsoleerd met "
        "<code>train_test_split(..., test_size=0.2, random_state=42, stratify=y)</code>. "
        "Stratificatie garandeert dat de 3,39% defectverhouding identiek vertegenwoordigd is in zowel train "
        "(271 defecten) als test (68 defecten). "
        "Cruciaal voor het voorkomen van informatielekkage: <b>alle schaling en encoders zitten binnen de "
        "Scikit-learn Pipeline</b>. De <code>StandardScaler</code> en <code>OneHotEncoder</code> worden "
        "uitsluitend gefit op de 8.000 trainingsrijen en pas daarna blind toegepast op de testdata. "
        "Er lekt geen enkele gemiddelde- of variantiewaarde van de testset door naar het model."
    )
    story.append(Paragraph("5. Data Leakage Risico & Eerlijke Train/Test-Split", h1_style))
    story.append(Paragraph(p5_a, body_style))
    story.append(Paragraph(p5_b, body_style))

    # Sectie 6
    p6 = (
        "In overeenstemming met de ML-engineeringprincipes uit les 5 bevat de testsuite "
        "(<code>tests/</code>, 12 geslaagde tests) geen breekbare punttests, maar valideert zij: "
        "<b>1. Vorm & type</b> (uitvoer is een 1D float array), "
        "<b>2. Invarianten</b> (alle voorspelde kansen liggen strikt tussen [0, 1]), "
        "<b>3. Pijplijnintegriteit</b> (ruwe data met overtollige kolommen passeert foutloos), "
        "<b>4. Persistentie</b> (model via <code>joblib</code> opslaan en herladen levert identieke kansen), "
        "<b>5. Reproduceerbaarheid</b> (vaste random state levert identiek model), en "
        "<b>6. Kwaliteitsondergrens</b> (het model faalt als test-recall zakt onder 0.60)."
    )
    story.append(Paragraph("6. Software- & Test-Invarianten", h1_style))
    story.append(Paragraph(p6, body_style))

    doc.build(story)


def main():
    cfg = laad_config()
    df = laad_data(cfg["data"]["pad"], valideer_met=None)

    plot_pad = PROJECT / "notebooks" / "modelvergelijking.png"
    pdf_pad = PROJECT / "Cloud_for_AI_Deel_2_Rapport.pdf"

    print("Genereren van evaluatiegrafieken...")
    genereer_grafiek(df, cfg, plot_pad)
    print(f"Grafiek opgeslagen als {plot_pad}")

    print("Genereren van het 2-pagina's PDF-rapport...")
    genereer_pdf(pdf_pad, plot_pad)
    print(f"PDF-rapport opgeslagen als {pdf_pad}")


if __name__ == "__main__":
    main()
