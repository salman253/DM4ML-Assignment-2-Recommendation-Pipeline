from pathlib import Path
import json, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

def generate_report(root):
    root=Path(root); reports=root/"reports"
    ev=pd.read_csv(reports/"evaluation_results.csv")
    val=json.loads((reports/"validation_report.json").read_text())
    summary=json.loads((reports/"run_summary.json").read_text())
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Body9",parent=styles["BodyText"],fontSize=9,leading=13,spaceAfter=6))
    story=[Paragraph("End-to-End Data Management Pipeline for a Recommendation System",styles["Title"]),
      Paragraph("Data Management for Machine Learning — Assignment 2",styles["Heading2"]),
      Paragraph("1. Dataset and objective",styles["Heading2"]),
      Paragraph("I implemented an end-to-end movie recommendation pipeline using the official MovieLens latest-small dataset published by GroupLens Research. It contains user ratings and movie metadata; user IDs are anonymized and no demographic attributes are provided. I use the ratings as user-item interactions and the movie genres/title as item metadata. Dataset source: https://grouplens.org/datasets/movielens/. I acknowledge Harper and Konstan (2015), The MovieLens Datasets: History and Context, ACM TiiS 5(4), DOI: 10.1145/2827872.",styles["Body9"]),
      Paragraph("2. Pipeline architecture and ingestion",styles["Heading2"])]
    arch=reports/"architecture.png"
    if arch.exists(): story.append(Image(str(arch),width=17*cm,height=5.4*cm))
    story += [Paragraph("I download and extract the GroupLens archive on first execution. I normalize ratings.csv into interactions, movies.csv into item metadata, and create a user table containing only anonymized IDs because MovieLens does not provide demographic details. Unix timestamps are converted to UTC. The validator checks required fields, duplicate user-item pairs, rating range and half-star increments, timestamp parsing and referential integrity. Invalid rows are quarantined; validation counts are shown below.",styles["Body9"]),
      Paragraph("Validation summary",styles["Heading3"]),
      Paragraph("Input rows: %s; valid rows: %s; invalid rows: %s."%(val["input_rows"],val["valid_rows"],val["invalid_rows"]),styles["Body9"]),
      Paragraph("3. Preparation and feature management",styles["Heading2"]),
      Paragraph("I sort interactions chronologically and reserve the latest 20%% as a test set, keeping the earlier 80%% for training. I export user activity and item popularity summaries, a user-item rating matrix in CSV and sparse NPZ formats, and reusable user, item and interaction feature tables. User features include activity count, average rating, preferred genre and days since last interaction. Item features include popularity, average rating, title and genres. Interaction features retain rating, UTC timestamp and event type.",styles["Body9"]),
      PageBreak(),
      Paragraph("4. Model training and evaluation",styles["Heading2"]),
      Paragraph("I compare a popularity baseline, which ranks movies by training rating count, with personalized item-item collaborative filtering. The personalized model computes cosine similarity between sparse movie rating vectors and scores candidates using a user's observed ratings. I save both trained models with joblib and record experiment parameters/artifacts locally; MLflow logging is enabled when the optional dependency is installed.",styles["Body9"]),
      Paragraph("The evaluation uses top-five recommendations. Precision@5 measures relevant recommendations per five slots; Recall@5 measures the fraction of held-out relevant movies recovered; NDCG@5 discounts lower-ranked hits; Coverage measures the fraction of catalog items recommended to at least one test user; Diversity is the proportion of unique genre labels in recommendation lists. RMSE and MAE are not applicable because the models rank items rather than predict numeric ratings.",styles["Body9"])]
    table=[list(ev.columns)]+[[str(r[c]) if c=="Model" else f"{r[c]:.4f}" for c in ev.columns] for _,r in ev.iterrows()]
    t=Table(table,repeatRows=1,colWidths=[6*cm]+[2.1*cm]*(len(ev.columns)-1))
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#315b7d")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("GRID",(0,0),(-1,-1),.4,colors.grey),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [t,Spacer(1,8),Paragraph("5. Orchestration, reproducibility and limitations",styles["Heading2"]),
      Paragraph("The Python workflow executes stages in dependency order from one command; an optional Airflow DAG invokes the same workflow. The split fraction and top-K setting are configurable. This dataset is for education/development and does not establish production performance. Limitations include anonymized users without demographic context, cold-start users/items, and the absence of online serving, drift monitoring and production retraining. MovieLens terms must be followed; this assignment uses the dataset for educational/non-commercial work.",styles["Body9"]),
      Paragraph("Run summary: %s users; %s movies; %s ratings; %s train rows; %s test rows."%(summary.get("users","n/a"),summary.get("items","n/a"),summary.get("raw_interactions","n/a"),summary.get("train_rows","n/a"),summary.get("test_rows","n/a")),styles["Body9"])]
    SimpleDocTemplate(str(reports/"final_report.pdf"),pagesize=A4,rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=1.4*cm,bottomMargin=1.4*cm).build(story)
