"""Render the analytical report from the notebook's checked output tables.

Run from the repository root after executing f1_podium_milestone1.ipynb.
The PDF and Markdown source share the same data and prose.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
HERE = Path(__file__).resolve().parent
SUMMARY = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
FIG = OUT / "figures"
TABLES = OUT / "tables"
HERE.mkdir(exist_ok=True)

NAVY = colors.HexColor("#18344d")
TEAL = colors.HexColor("#1d6b79")
GRAY = colors.HexColor("#5d6772")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
    fontSize=21, leading=25, textColor=NAVY, spaceAfter=14))
styles.add(ParagraphStyle(name="ReportH1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=14, leading=18, textColor=NAVY, spaceBefore=15, spaceAfter=7, keepWithNext=1))
styles.add(ParagraphStyle(name="ReportH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=10.5, leading=14, textColor=TEAL, spaceBefore=9, spaceAfter=5, keepWithNext=1))
styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=9, leading=13, spaceAfter=7))
styles.add(ParagraphStyle(name="ReportSmall", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=7.3, leading=10, textColor=GRAY, spaceAfter=4))
styles.add(ParagraphStyle(name="ReportCaption", parent=styles["BodyText"], fontName="Helvetica-Oblique",
    fontSize=7.8, leading=10.5, textColor=GRAY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10))

story = []
markdown = []

def p(text):
    story.append(KeepTogether([Paragraph(html.escape(text), styles["ReportBody"])]))
    markdown.extend([text, ""])

def h1(text):
    story.append(Paragraph(html.escape(text), styles["ReportH1"]))
    markdown.extend([f"## {text}", ""])

def h2(text):
    story.append(Paragraph(html.escape(text), styles["ReportH2"]))
    markdown.extend([f"### {text}", ""])

def fig(name, caption, max_height=8*cm):
    path = FIG / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    im = Image(str(path))
    max_width = 17.2*cm
    ratio = min(max_width/im.imageWidth, max_height/im.imageHeight)
    im.drawWidth = im.imageWidth*ratio
    im.drawHeight = im.imageHeight*ratio
    story.extend([im, Paragraph(html.escape(caption), styles["ReportCaption"])])
    markdown.extend([f"![{caption}](../outputs/figures/{name}.png)", ""])

def table(headers, rows, widths=None):
    header_style=ParagraphStyle(name="TableHeader", parent=styles["ReportSmall"],
                                textColor=colors.white, fontName="Helvetica-Bold")
    data = [[Paragraph(html.escape(str(v)), header_style) for v in headers]]
    data += [[Paragraph(html.escape(str(v)), styles["ReportSmall"]) for v in row] for row in rows]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),NAVY),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
        ("GRID",(0,0),(-1,-1),.2,colors.HexColor("#d9e1e5")),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    story.extend([t,Spacer(1,6)])
    markdown.append("| " + " | ".join(str(x) for x in headers) + " |")
    markdown.append("| " + " | ".join("---" for _ in headers) + " |")
    markdown.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    markdown.append("")

def page():
    story.append(Spacer(1,0.4*cm))

def pct(value, digits=1):
    return f"{100*float(value):.{digits}f}%"

def num(value, digits=3):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"

def top_line(rows):
    if not rows:
        return "No circuit met the minimum sample rule."
    return "; ".join(f"{r['circuit']} {pct(r['rate'])} ({r['starts']} front-row starts)" for r in rows)

story.append(Paragraph("Formula 1 Podium Prediction", styles["ReportTitle"]))
story.append(Paragraph("Milestone 1 analytical report | 1950-2024 source, 2003-2024 modeling",
                       styles["ReportBody"]))
markdown.extend(["# Formula 1 Podium Prediction", "Milestone 1 analytical report | 1950-2024 source, 2003-2024 modeling", ""])

h1("Executive summary")
m = SUMMARY["modeling"]
comparison = pd.DataFrame(m["comparison"])
winner = m["winner"]
test = comparison.query("model == @winner and split == 'test'").iloc[0]
p(f"A driver-race starter is predicted to finish on the podium using only data available after qualifying and before the race. "
  f"The selected {winner} scored test PR-AUC {num(test.pr_auc)}, ROC-AUC {num(test.roc_auc)}, "
  f"F1 {num(test.f1)}, precision {num(test.precision)}, and recall {num(test.recall)} on unseen 2022-2024 races. "
  "PR-AUC is primary because podiums are uncommon; the F1 threshold was chosen on 2020-2021 validation data.")
p("This report reproduces the notebook's three relational analyses, audit and cleaning evidence, feature rationale, "
  "all model comparisons, controlled experiments, model explanations and limitations. The notebook is the executable source of truth.")

h1("1. Data audit and cleaning")
d=SUMMARY["dataset"]; c=SUMMARY["cleaning"]
p(f"The immutable source contains 14 linked CSV tables and {d['raw_results']:,} results rows across "
  f"{d['years'][0]}-{d['years'][1]}. Literal missing markers were converted to nulls; dates, numeric columns and "
  "lap/pit durations were typed, with parse failures and key checks recorded in the notebook. "
  f"Qualifying coverage was {pct(d['qualifying_2002'])} in 2002 and {pct(d['qualifying_2003'])} in 2003, "
  "supporting the 2003 modeling start. Primary keys were checked for uniqueness and non-null values; foreign keys for orphans.")
p(f"The cleaning rules removed {c['indy_removed']:,} Indianapolis 500 rows, {c['nonstarters_removed']:,} non-starters, "
  f"and {c['duplicates_removed']:,} extra shared-drive rows, leaving {c['starts']:,} unique driver-race starts. "
  "A unique positive-grid started car wins a shared-drive tie; otherwise the best finishing result wins. "
  "The cleaned podium rate is " + pct(c["podium_rate"]) + f". In {c['historical_multi_podium_races']} "
  "early shared-drive races, more than three distinct drivers still have a top-three positionOrder; the required "
  "target definition is retained. The raw files were never changed.")
fig("02_raw_era_eda","Figure 1. Historical entries, raw podium prevalence and qualifying coverage.")
fig("04_cleaning_impact","Figure 2. The measured impact of cleaning on rows, class balance and missingness.")

page(); h1("2. Three data-engineering answers")
h2("Q1. Which circuits convert a front-row start into a podium?")
q1=SUMMARY["questions"]["q1"]
for era in ["pre-2014","hybrid 2014-2021","ground effect 2022-2024"]:
    p(f"Top eligible circuits in {era}: {top_line(q1['top_by_era'][era])}.")
rank_rows=[]
for r in q1["rank_change"]:
    rank_rows.append([r["period_a"],r["period_b"],r["shared_circuits"],num(r["spearman_rho"],2)])
table(["From","To","Shared circuits","Spearman rho"],rank_rows,
      widths=[4.2*cm,4.6*cm,2.7*cm,3.4*cm])
change=q1["qualifying_grid"]
qrows=[[r["era"],r["qualified_rows"],r["qualifying_front_row"],r["grid_front_row"],r["changed_start"]]
       for r in change]
table(["Era","Qualifying rows","Qual front row","Grid front row","Grid differs"],qrows,
      widths=[4.0*cm,3*cm,3.1*cm,3.1*cm,2.4*cm])
p("The unit is a driver who actually started from grid 1 or 2. Rates use final podium outcomes only for the answer, "
  "never as pre-race predictors. The rank comparisons require ten front-row starts before 2022 and four in 2022-2024; "
  "wide intervals limit claims about the 2022-2024 ranking. Qualifying position is compared separately where recorded. "
  "The pre-2014 grid analysis spans all available historical starts; qualifying-only comparisons have changing coverage.")
p("The shared-circuit rank correlations in the table are near zero or negative, so these observed rankings "
  "do change across eras. The small recent samples and overlap of intervals prevent a strong claim that "
  "individual circuits themselves became inherently more or less favorable.")
fig("05_q1_front_row_by_era","Figure 3. Front-row conversion by circuit and era; error bars are Wilson 95% intervals.")

h2("Q2. Is a home race associated with more podiums after accounting for grid?")
q2=SUMMARY["questions"]["q2"]
p(f"Among {q2['analyzed_starts']:,} classifiable starts ({q2['excluded_ambiguous_nationality']:,} with ambiguous nationality "
  f"excluded), {q2['home_starts']:,} were home starts. Adjusting for grid bucket and era, the home indicator's "
  f"odds ratio was {num(q2['odds_ratio'],2)} (race-clustered 95% CI {num(q2['or_low'],2)}-"
  f"{num(q2['or_high'],2)}). Standardized model probabilities were "
  f"{pct(q2['adjusted_home_probability'])} at home and {pct(q2['adjusted_away_probability'])} away.")
p("The confidence interval includes an odds ratio of 1, so these data do not show a clear home-podium advantage "
  "after the stated grid and era adjustment.")
p("The unit is one cleaned driver-race start. British/UK and American/USA, among other demonyms, were mapped "
  "explicitly; mixed or unmapped nationalities were excluded. Grid strata control for starting position and era "
  "adjusts a major historical difference. Team strength, circuit familiarity and driver selection may still confound "
  "the comparison, so the association is not causal.")
fig("06_q2_home_by_grid","Figure 4. Home and away podium rates within grid groups, with Wilson 95% intervals.")

page(); h2("Q3. Which constructors retire mechanically most often?")
q3=SUMMARY["questions"]["q3"]
rows=[]
for era in ["2014-2021","2022-2024"]:
    for r in q3["top_by_era"][era]:
        rows.append([era,r["constructor"],f"{r['mechanical']}/{r['starts']}",pct(r["rate"]),
                     f"{pct(r['low'])}-{pct(r['high'])}",r["ambiguous"],pct(r["sensitivity_upper"])])
table(["Era","Constructor","Failures/starts","Rate","95% CI","Ambig.","Upper rate"],rows,
      widths=[2.2*cm,3.3*cm,2.7*cm,1.8*cm,2.8*cm,1.5*cm,2.2*cm])
p(f"There were {q3['ambiguous_starts']} ambiguous-status starts across 2014-2024. The main numerator includes "
  "explicitly named car-component and subsystem failures; collision, driver, regulatory and generic/uncertain "
  "retirements are excluded. Every actual race start remains in the denominator. Rankings require at least "
  "20 starts per constructor-era; the notebook's status mapping and sensitivity upper rate show how ambiguous "
  "labels could change a result. Constructor IDs remain distinct across renames.")
fig("07_q3_mechanical_by_era","Figure 5. Highest mechanical-retirement rates by constructor; Wilson 95% intervals.")

page(); h1("3. Features, cutoff and validation design")
p("The modeling grain is one unique (raceId, driverId) starter from 2003-2024. Grid and qualifying fields are fixed "
  "before the main race. Driver form uses the previous five starts; constructor form first aggregates each previous "
  "race, then shifts by one constructor-race. Standings are taken from the latest strictly earlier round of the same "
  "season. Age, home indicator, race round and qualifying-entry count are pre-race context. Current-race results, "
  "status, points, laps, pit stops, same-race standings and sprint outcomes are excluded as features.")
feature_groups=m["feature_groups"]
feature_why=pd.read_csv(TABLES/"feature_justification.csv").set_index("group")
table(["Feature group","Fields","Pre-race rationale"],
      [[group,", ".join(fields),feature_why.loc[group,"reason"]] for group,fields in feature_groups.items()],
      widths=[2.5*cm,7.0*cm,7.0*cm])
splits=pd.read_csv(TABLES/"split_summary.csv")
table(["Split","Years","Rows","Races","Podium rate"],
      [[r["split"],r["years"],r["rows"],r["races"],pct(r["podium_rate"])] for _,r in splits.iterrows()],
      widths=[3*cm,3*cm,3*cm,3*cm,3.8*cm])
p("No random split mixes future races into training. Imputation, clipping, log transformation and scaling are fitted "
  "on the training seasons only. Earlier held-out race outcomes may become legitimate history for later races; "
  "no held-out label changes model parameters or the chosen threshold.")
fig("08_feature_evidence","Figure 6. Training-only evidence for grid, driver form and constructor form.")
fig("10_feature_missingness","Figure 7. Missing training features that require fitted imputation.")

page(); h1("4. Models and controlled experiments")
p(f"The simple baseline is logistic regression. Random forest, histogram gradient boosting and a one-hidden-layer "
  f"feedforward neural network use the same rows and feature list. A small declared parameter grid is visible in "
  f"the notebook; each family's best variant and its F1 threshold were selected on validation data. The overall "
  f"winner was {winner}. Test metrics were computed only after those choices were fixed.")
p("Logistic regression fits linear log odds and may miss interactions; it provides an interpretable baseline. "
  "Random forest averages bagged trees but can overfit. Gradient boosting adds trees sequentially and is "
  "sensitive to tuning and era drift. The one-hidden-layer neural network learns nonlinear combinations "
  "through backpropagation but needs scaling and can converge slowly. The shared split tests whether "
  "their extra flexibility improves ranking enough to justify complexity.")
if m.get("nn_epoch_cap_reached"):
    p("The selected neural-network variant reached its declared 160-epoch cap. Its validation and test scores "
      "are reported without claiming full optimizer convergence; the training-loss curve is included in the notebook.")
metric_rows=[]
for _,r in comparison.iterrows():
    metric_rows.append([r["model"].replace("Shallow neural network","Shallow NN"),r["split"],
       num(r["roc_auc"]),num(r["pr_auc"]),num(r["f1"]),num(r["precision"]),num(r["recall"])])
table(["Model","Split","ROC-AUC","PR-AUC","F1","Precision","Recall"],metric_rows,
      widths=[4.2*cm,2.3*cm,2.1*cm,2.1*cm,1.9*cm,2.2*cm,2.2*cm])
fig("13_model_curves_test","Figure 8. Final ROC and precision-recall comparisons on unseen test seasons.")

h2("Preprocessing order")
orders=pd.DataFrame(m["preprocessing_order"]["trials"])
table(["Experiment","Log before impute","Clip before impute","Validation PR-AUC","Validation F1"],
      [[r["experiment"],r["log_before_impute"],r["clip_before_impute"],num(r["pr_auc"]),num(r["f1"])]
       for _,r in orders.iterrows()],widths=[3.4*cm,3.4*cm,3.4*cm,3.5*cm,3.0*cm])
p("The log/mean experiment changes how missing previous-points counts are filled. The clipping/mean experiment "
  "changes whether outliers affect the imputation mean. Both use a fixed logistic baseline and identical columns "
  "and rows. The winning orders were selected by validation PR-AUC, with all fitted quantities derived from train.")
fig("11_preprocessing_raw_distributions","Figure 9. Example raw feature distributions before transformation.")
fig("12_preprocessing_transformed_distributions","Figure 10. Corresponding transformed training distributions.")

page(); h2("Feature-group ablation")
ab=pd.DataFrame(m["ablation"])
table(["Removed group","Features left","Validation PR-AUC","F1","PR-AUC drop"],
      [[r["removed"],r["features"],num(r["pr_auc"]),num(r["f1"]),num(r["delta_pr_auc"])]
       for _,r in ab.iterrows()],widths=[4.2*cm,2.9*cm,3.4*cm,2.2*cm,3.7*cm])
p("Each row refits the selected family with the same settings and removes one group at a time. A negative "
  "drop means validation ranking improved when the group was removed; this is evidence about this fitted "
  "model, not a universal claim about that information.")
largest=ab.iloc[1:].sort_values("delta_pr_auc",ascending=False).iloc[0]
p(f"Removing {largest['removed']} caused the largest validation PR-AUC decrease ({num(largest['delta_pr_auc'])}). "
  "The ablation plot also makes any redundant or harmful group visible; a near-zero change is not proof "
  "that the underlying information is unimportant in all models.")
fig("16_feature_ablation","Figure 11. Validation PR-AUC change after retraining without each group.")

h1("5. Global and local explanations")
x=SUMMARY["xai"]
perm=", ".join(r["feature"] for r in x["top_permutation"][:3])
shap=", ".join(r["feature"] for r in x["top_shap"][:3])
p(f"Permutation importance ranks {perm} highest by validation PR-AUC decrease. Mean absolute SHAP values on a "
  f"bounded validation sample highlight {shap}. These methods can split importance differently among correlated "
  "grid, qualifying, standings and form fields. Neither establishes that changing a field causes a podium.")
fig("17_permutation_importance","Figure 12. Global permutation importance for the selected model.")
fig("18_shap_global","Figure 13. Global mean absolute SHAP values on validation rows.")

page(); h2("One prediction")
p(f"The local example is race {x['local_race_id']}, driver {x['local_driver_id']}, with actual podium label "
  f"{x['local_actual_podium']} and predicted probability {pct(x['local_probability'])}. The SHAP waterfall "
  "attributes movement from a training-background expectation to this prediction; LIME fits a local surrogate "
  "near the same input. The examples describe this model prediction, not race mechanics or causal effects.")
p("For this case, SHAP's largest upward contributions are " + ", ".join(x["local_shap_up"]) +
  "; the largest downward contributions are " + ", ".join(x["local_shap_down"]) +
  ". The LIME bars show the corresponding local surrogate coefficients. Their values depend on the chosen "
  "background or local perturbations, so they should be read as explanations of this fitted prediction.")
fig("20_shap_local","Figure 14. Local SHAP waterfall for one unseen driver-race.")
fig("21_lime_local","Figure 15. LIME's local approximation around the same pre-race input.")

h1("6. Inference and limits")
inf=m["inference"]
p(f"The notebook's predict_podium(record) accepts only named engineered pre-race fields and applies the frozen "
  f"training-fitted preprocessing and model, plus the validation threshold. A complete record returned probability "
  f"{pct(inf['complete']['podium_probability'])} and label {inf['complete']['predicted_podium']}; "
  f"with qualifying position, grid change and previous driver points omitted, it returned "
  f"{pct(inf['partial']['podium_probability'])} and label {inf['partial']['predicted_podium']}. "
  "An unexpected or post-race field is rejected.")
p("Limitations include pre-2003 qualifying drift, rule and car changes between train/validation and the 2022-2024 "
  "test era, constructor-identity resets at renames, uncertain mechanical-status labels, small home/front-row "
  "subgroups, class imbalance and the fact that the model is evaluated on actual starters. A genuine future "
  "deployment must reconstruct history only from races already completed at its cutoff.")
p("Source: Rohan Rao, Formula 1 World Championship (1950-2024), Kaggle/Ergast, "
  "https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020; milestone brief and the "
  "companion Formula 1 introduction supplied with this project. The full notebook, figures, data mapping, "
  "parameter attempts and output tables provide the reproducible evidence for this report.")

def footer(canvas, doc):
    canvas.saveState()
    w,h=A4
    canvas.setStrokeColor(colors.HexColor("#d4dee4"))
    canvas.line(1.8*cm,1.4*cm,w-1.8*cm,1.4*cm)
    canvas.setFont("Helvetica",7)
    canvas.setFillColor(GRAY)
    canvas.drawString(1.8*cm,1.05*cm,"Formula 1 Podium Prediction | Milestone 1")
    canvas.drawRightString(w-1.8*cm,1.05*cm,f"Page {doc.page}")
    canvas.restoreState()

doc=SimpleDocTemplate(str(HERE/"report.pdf"),pagesize=A4,
    leftMargin=1.8*cm,rightMargin=1.8*cm,topMargin=1.7*cm,bottomMargin=1.8*cm,
    title="Formula 1 Podium Prediction - Milestone 1 Analytical Report",
    author="ACL Project")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
(HERE/"report.md").write_text("\n".join(markdown).rstrip()+"\n",encoding="utf-8")
print("Created",HERE/"report.pdf","and",HERE/"report.md")
