import argparse
import numpy as np
import torch
import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd
from rapidfuzz import fuzz

from gliner import GLiNER

from tqdm.auto import tqdm

# -----------------------------
# 0. Input Arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-path",
    required=True,
    help="Path or Hugging Face model identifier for the GLiNER model.",
)
args = parser.parse_args()

# -----------------------------
# 1. Ground truth for your 64 examples
# -----------------------------

GOLD = [
    {
        "title": 'Gee Di Moda Rectangle Tablecloth - 60 x 126" Inch - Black Rectangular Table Cloth for 8 Foot Table in Washable Polyester - Great for Buffet Table, Parties, Holiday Dinner, Wedding & More',
        "aliases": ["tablecloth", "rectangle tablecloth", "rectangular tablecloth", "table cloth", "rectangular table cloth"],
    },
    {
        "title": "Wedding Favors for Guests Party Favors Rustic Vintage Key Bottle Opener with Escort Card Tag and Twine 50 Pcs",
        "aliases": ["bottle opener", "key bottle opener"],
    },
    {
        "title": "Tree of Life Metal Wall Art, Contemporary Iron Artwork Decor, Celtic Family Trees, Round Modern Plaque, Handmade in Haiti,Fair Trade Certified",
        "aliases": ["wall art", "metal wall art", "artwork decor", "plaque", "wall plaque"],
    },
    {
        "title": 'Valery Madelyn 48" Trendy Red Tartan Plaid Christmas Tree Skirt, Themed with Christmas Ornaments (Not Included)',
        "aliases": ["christmas tree skirt", "tree skirt"],
    },
    {
        "title": "Willow Tree hand-painted sculpted figure, Shine",
        "aliases": ["sculpted figure", "figure", "figurine"],
    },
    {
        "title": "Shot Glasses Premium 1oz Clear Plastic Disposable Cups 500 ct VALUE PACK, Perfect Container for Jello Shots, Condiments, Tasting, Sauce, Dipping, Samples",
        "aliases": ["shot glasses", "shot glass", "disposable cups", "plastic cups", "cups"],
    },
    {
        "title": "Pelican 1750 Rifle Case with Foam (Long Case, Multi-Purpose) - Desert Tan",
        "aliases": ["rifle case", "case"],
    },
    {
        "title": "Step2 Up & Down Roller Coaster Ride On",
        "aliases": ["roller coaster", "ride on", "ride on toy", "roller coaster ride on"],
    },
    {
        "title": "Learning Resources Garden Fresh Salad Set, 38 Pieces",
        "aliases": ["salad set", "play food set", "toy food set"],
    },
    {
        "title": 'shock cord - marine grade. 1/8", 3/16", 1/4" diameter, 25 / 50 / 100 ft. 6 colors. includes 2 carabiners & knot tying ebook. made in usa. also called bungee cord, stretch cord & elastic cord.',
        "aliases": ["shock cord", "bungee cord", "stretch cord", "elastic cord"],
    },
    {
        "title": "Fisher-Price Imaginext Shark Bite Pirate Ship",
        "aliases": ["pirate ship", "toy pirate ship"],
    },
    {
        "title": "Canon EF 100mm f/2.8L IS USM Macro Lens for Canon Digital SLR Cameras",
        "aliases": ["macro lens", "lens", "camera lens"],
    },
    {
        "title": "xl hammock straps - hammock tree straps set versatile 2000+ lbs heavy duty 24-40 loops & 100% no stretch suspension system kit - camping hammock accessories | 2 carabiners | 10 or 12 feet options",
        "aliases": ["hammock straps", "hammock tree straps", "tree straps", "suspension system kit"],
    },
    {
        "title": "Willow Tree Creche",
        "aliases": ["creche", "nativity creche"],
    },
    {
        "title": "Hummingbird Glass Straws Caramel Frog on Clear Bent 9 in x 9.5 mm With Cleaning Brush",
        "aliases": ["glass straws", "straws"],
    },
    {
        "title": "LEGO Ideas 21301 Birds Model Kit(Discontinued by manufacturer)",
        "aliases": ["model kit", "birds model kit", "building kit"],
    },
    {
        "title": '18 Inch Doll Clothes/Clothing Fits American Girl Doll - Fur Collar Accessory Jacket Outfit with White T-Shirt and Distressed Jeans',
        "aliases": ["doll clothes", "doll clothing", "doll outfit", "jacket outfit"],
    },
    {
        "title": "Elements Friend Angel Figurine by Pavilion, 8-Inch, Holding Butterfly, Inscription Friends Are Gifts to Always Treasure",
        "aliases": ["angel figurine", "figurine"],
    },
    {
        "title": "Ukulele Concert Size Bundle From Lohanu (LU-C) 2 Strap Pins Installed FREE Uke Strap Case Tuner Picks Pick Holder Aquila Strings Installed Free Video Lessons BEST UKULELE BUNDLE DEAL Purchase Today...",
        "aliases": ["ukulele", "ukulele bundle", "concert ukulele", "ukulele kit"],
    },
    {
        "title": "Evergreen Garden Dog Paw in Hand Devotion Painted Polystone Stepping Stone - 12”W x 0.5”D x 7.5”H",
        "aliases": ["stepping stone", "garden stepping stone"],
    },
    {
        "title": "LEGO Friends Jungle Bridge Rescue 41036 Building Set",
        "aliases": ["building set", "lego building set", "toy building set"],
    },
    {
        "title": "Safavieh Hudson Shag Collection SGH282A Ivory and Grey Area Rug, 6 feet by 9 feet (6' x 9')",
        "aliases": ["area rug", "rug"],
    },
    {
        "title": "Willow Tree hand-painted sculpted angel, Angel's Embrace",
        "aliases": ["sculpted angel", "angel figurine", "figurine", "angel"],
    },
    {
        "title": "StarPack Basics Range Silicone Kitchen Tongs 2 Pack (9-Inch & 12-Inch) in FDA Grade, Bonus 101 Cooking Tips (Cherry Red)",
        "aliases": ["kitchen tongs", "tongs"],
    },
    {
        "title": "The Rooster Mason Jar Shot Glasses Set - Mini Mason Jars with Lids Featuring Unique Rooster Design (10 Pack)",
        "aliases": ["shot glasses set", "shot glasses", "mason jar shot glasses", "mini mason jars"],
    },
    {
        "title": "Flexifoil 2.05m Power Kite, Big Buzz Sport Foil - Safe, Reliable, Durable Family Orientated Power Kiting",
        "aliases": ["power kite", "kite", "sport foil"],
    },
    {
        "title": "BTäT - Tea Set, China Tea Set, Tea Service, Tea Cups (8oz), Creamer and Sugar Set, Tea Cups and Saucer Set, Tea Pot, Tea Set for Adults, Tea Cups Set of 4, Blue Dream Tea Set, Porcelain Tea Set, Cups",
        "aliases": ["tea set", "china tea set", "tea service", "porcelain tea set"],
    },
    {
        "title": "18 Inch Doll Horse Riding Outfit, 5 Piece Complete Navy Equestrian Set fits 18 Inch American Girl Dolls & More! Includes Boots and Helmet | Gift Bag Included",
        "aliases": ["doll outfit", "doll riding outfit", "horse riding outfit", "equestrian set"],
    },
    {
        "title": "A+ Book Stand BS1500 Book Holder w/ Adjustable Foldable Tray and Page Paper Clips-Cookbook Reading Desk Portable Sturdy Lightweight Bookstand-Textbooks Bookstands-Music Books Tablet Cook Recipe Stands",
        "aliases": ["book stand", "book holder", "bookstand", "reading stand"],
    },
    {
        "title": "1InTheHome Heavy Duty White Hangers Tubular Plastic Hangers, Set of 24 (Heavy Duty)",
        "aliases": ["hangers", "clothes hangers", "plastic hangers"],
    },
    {
        "title": "Learning Resources New Sprouts Cookout! Food, 19 Pieces",
        "aliases": ["play food", "toy food", "food set", "play food set"],
    },
    {
        "title": "Willow Tree Hand-Painted Sculpted Figure, Kindness (Girl)",
        "aliases": ["sculpted figure", "figure", "figurine"],
    },
    {
        "title": "OXO Good Grips Baker’s Dusting Wand for Sugar, Flour and Spices",
        "aliases": ["dusting wand", "baker's dusting wand"],
    },
    {
        "title": "misslo durable hanging closet double sided bra stocking clothes socks organizer 15 large mesh pockets, white",
        "aliases": ["closet organizer", "hanging organizer", "bra organizer", "socks organizer", "clothes organizer"],
    },
    {
        "title": "Pennsylvania Woodworks Extra Large Wooden Clothes Drying Rack",
        "aliases": ["clothes drying rack", "drying rack"],
    },
    {
        "title": "canopy street adorable animal thank you note card assortment pack",
        "aliases": ["thank you cards", "note cards", "thank you note cards", "note card assortment pack"],
    },
    {
        "title": "Pelican 1720 Rifle Case with Foam (Long Case, Multi-Purpose) - Black",
        "aliases": ["rifle case", "case"],
    },
    {
        "title": "Culina Stackable 6-Piece Stainless Steel Measuring Cups",
        "aliases": ["measuring cups", "measuring cup set"],
    },
    {
        "title": "10lbs bulk non-gmo dried mealworms for reptile , tortoise ; amphibian ,lizard ;wild birds; chichens; duck etc",
        "aliases": ["dried mealworms", "mealworms"],
    },
    {
        "title": "mio marino winter cashmere feel women scarf, 100% cotton fashion scarves, in elegant gift box - gallant - one size",
        "aliases": ["women scarf", "scarf", "scarves", "fashion scarf"],
    },
    {
        "title": "LEGO City Fire Station 60004",
        "aliases": ["fire station", "toy fire station", "building set", "lego building set"],
    },
    {
        "title": "Hyperikon LED 70W Wall Pack Fixture, 350-400W HPS/HID Replacement, 5000K, 6000 Lumens, Waterproof and Outdoor Rated, UL & DLC, Shield Included",
        "aliases": ["wall pack fixture", "led wall pack fixture", "light fixture"],
    },
    {
        "title": "stainless steel cocktail muddler and mixing spoon set by varvino with a! - make flavorful cocktails with ease!",
        "aliases": ["cocktail muddler", "muddler", "mixing spoon set", "muddler and spoon set", "cocktail muddler and mixing spoon set"],
    },
    {
        "title": "Willow Tree Mother Daughter Figurine",
        "aliases": ["figurine", "mother daughter figurine"],
    },
    {
        "title": 'Emily Rose Doll Clothes 18 Inch Doll Clothes/Clothing Fits 18" American Girl Dolls - Value Pack Plain White T-shirts 18" Outfit I Gift-Boxed!',
        "aliases": ["doll clothes", "doll clothing", "doll outfit", "doll t-shirts"],
    },
    {
        "title": "LEGO City Police 60047 Police Station",
        "aliases": ["police station", "toy police station", "building set", "lego building set"],
    },
    {
        "title": "sertodo copper cmm-18 moscow mule mug, hand hammered 100% pure copper, 18 oz, single",
        "aliases": ["moscow mule mug", "copper mug", "mug"],
    },
    {
        "title": "Luxury Moscow Mule Copper Mugs - Set of 2 in Gift box, 16 Oz barrel / Cup - 100% Pure Solid Original Handmade and Hammered - Bonus : Two Stainless Steel Stirring Spoons, Recipe E-book. Top Quality",
        "aliases": ["moscow mule mugs", "copper mugs", "mugs"],
    },
    {
        "title": "Dog Art Wood Tinket Box -Original Work of Wood Art -Unmatched Quality -Handmade in USA -Unique, No Two Are the Same. Golden Retriever Dog Picture Makes a Great Gift for the Dog Lover in Your Life.",
        "aliases": ["trinket box", "wood trinket box", "box"],
    },
    {
        "title": "Green Toys DIN01R Cookware and Dining Set",
        "aliases": ["cookware and dining set", "cookware set", "dining set", "play cookware set"],
    },
    {
        "title": "Malamute Rugged AA Battery Storage Case - Holds 8, Traction Feet, Made in the USA",
        "aliases": ["battery storage case", "storage case", "battery case"],
    },
    {
        "title": "MEMORIALS 4U Memorials4u Elite Cloud Blue and Silver Cremation Urn for Human Ashes - Adult Funeral Urn Handcrafted - Affordable Urn for Ashes - Large Urn Deal.",
        "aliases": ["cremation urn", "funeral urn", "urn"],
    },
    {
        "title": "Orca Tactical MOLLE EMT Medical First Aid IFAK Utility Pouch (Bag Only)",
        "aliases": ["utility pouch", "medical pouch", "first aid pouch", "ifak pouch", "bag"],
    },
    {
        "title": "RC 6 Pairs XT60 XT-60 Male Female Bullet Connectors Power Plugs with Heat Shrink for RC Lipo Battery by Finware",
        "aliases": ["bullet connectors", "connectors", "power plugs", "xt60 connectors"],
    },
    {
        "title": "LEGO Friends Jungle Rescue Base Building Set 41038",
        "aliases": ["building set", "lego building set", "toy building set"],
    },
    {
        "title": "LEGO Coast Guard Helicopter",
        "aliases": ["helicopter", "toy helicopter", "building set", "lego building set"],
    },
    {
        "title": "trunk crate pro collapsible portable multi compartments heavy duty non-slip cargo trunk organizer storage, blue",
        "aliases": ["trunk organizer", "cargo trunk organizer", "organizer"],
    },
    {
        "title": "mDesign Refrigerator and Freezer Storage Organizer Bins for Kitchen, 4 pc Set - Clear",
        "aliases": ["storage organizer bins", "organizer bins", "refrigerator bins", "freezer bins"],
    },
    {
        "title": "Ashcroft Twist Whiskey Glass Set or 2, Unique Modern Rocks Lead Free Crystal Glasses for Scotch or Bourbon With Luxury Gift Box",
        "aliases": ["whiskey glass set", "rocks glasses", "whiskey glasses", "glass set"],
    },
    {
        "title": 'Valery Madelyn 48" Traditional Red Green Gold Christmas Tree Skirt with Ruffle Trim Border, Themed with Christmas Ornaments (Not Included)',
        "aliases": ["christmas tree skirt", "tree skirt"],
    },
    {
        "title": 'Streamlight 44931 Siege Compact, Rugged 7.25" Hand Lantern 540 Lumen Uses 3D Cell Alkaline Batteries - 540 Lumens',
        "aliases": ["hand lantern", "lantern"],
    },
    {
        "title": 'Crescent Moon Wall Sculpture, Outdoor Home Decor, Recycled Metal Art, 15" X 15"',
        "aliases": ["wall sculpture", "wall art", "metal art"],
    },
    {
        "title": "WERTHER'S ORIGINAL Caramel Hard Candy, Bulk Candy, Hard Candy, Individually Wrapped Candy, Caramel Candy, Caramel Sweets, 34 Ounce Bags (Pack of 2)",
        "aliases": ["hard candy", "caramel hard candy", "candy", "caramel candy"],
    },
    {
        "title": "321 STRONG Refillable Chalk Ball with 65 Gram (2.3 oz) Capacity, Comes Full - for Rock Climbing, Gym Workouts, Billiards, and More",
        "aliases": ["chalk ball", "refillable chalk ball"],
    },
]


# -----------------------------
# 2. Eval-only normalization and scoring
# -----------------------------

def normalize_for_eval(s: str) -> str:
    """
    Evaluation-only normalization.
    This is NOT applied before GLiNER and does not improve predictions.
    """
    if s is None:
        return ""

    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_counter(s: str) -> Counter:
    return Counter(normalize_for_eval(s).split())


def token_f1(pred: str, gold: str) -> float:
    """
    Penalizes extra tokens, unlike token_set_ratio.
    Example:
      pred='Valery Madelyn Christmas Tree Skirt'
      gold='Christmas Tree Skirt'
    gets partial credit, not full credit.
    """
    p = token_counter(pred)
    g = token_counter(gold)

    if not p or not g:
        return 0.0

    common = sum((p & g).values())
    precision = common / sum(p.values())
    recall = common / sum(g.values())

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def best_alias_match(pred_text: str, aliases: List[str]) -> Dict[str, Any]:
    best = {
        "best_alias": None,
        "token_f1": 0.0,
        "token_set_ratio": 0.0,
        "exact_norm": False,
    }

    pred_norm = normalize_for_eval(pred_text)

    for alias in aliases:
        alias_norm = normalize_for_eval(alias)
        f1 = token_f1(pred_text, alias)
        tsr = fuzz.token_set_ratio(pred_norm, alias_norm)
        exact = pred_norm == alias_norm

        if f1 > best["token_f1"]:
            best = {
                "best_alias": alias,
                "token_f1": f1,
                "token_set_ratio": tsr,
                "exact_norm": exact,
            }

    return best


def score_predictions_for_one_title(
    preds: List[Dict[str, Any]],
    aliases: List[str],
    target_labels: Optional[List[str]] = None,
    strict_f1_cutoff: float = 0.80,
) -> Dict[str, Any]:
    """
    Scores target-label predictions for one title.

    selected_* = highest-confidence target-label prediction.
    oracle_* = best target-label prediction by gold match.
    """
    if preds is None:
        preds = []

    if target_labels is not None:
        target_preds = [p for p in preds if p.get("label") in target_labels]
    else:
        target_preds = list(preds)

    if not target_preds:
        return {
            "n_target_preds": 0,
            "selected_text": None,
            "selected_label": None,
            "selected_model_score": None,
            "selected_best_alias": None,
            "selected_token_f1": 0.0,
            "selected_token_set_ratio": 0.0,
            "selected_correct_strict": False,
            "oracle_text": None,
            "oracle_label": None,
            "oracle_best_alias": None,
            "oracle_token_f1": 0.0,
            "oracle_token_set_ratio": 0.0,
            "oracle_correct_strict": False,
        }

    # Highest model score = what you'd likely choose without postprocessing
    selected = max(target_preds, key=lambda p: float(p.get("score", 0.0)))
    selected_match = best_alias_match(selected["text"], aliases)

    # Best possible among returned target predictions = useful diagnostic
    oracle_rows = []
    for p in target_preds:
        m = best_alias_match(p["text"], aliases)
        oracle_rows.append((p, m))

    oracle_pred, oracle_match = max(oracle_rows, key=lambda x: x[1]["token_f1"])

    return {
        "n_target_preds": len(target_preds),

        "selected_text": selected.get("text"),
        "selected_label": selected.get("label"),
        "selected_model_score": float(selected.get("score", 0.0)),
        "selected_best_alias": selected_match["best_alias"],
        "selected_token_f1": selected_match["token_f1"],
        "selected_token_set_ratio": selected_match["token_set_ratio"],
        "selected_correct_strict": selected_match["token_f1"] >= strict_f1_cutoff,

        "oracle_text": oracle_pred.get("text"),
        "oracle_label": oracle_pred.get("label"),
        "oracle_best_alias": oracle_match["best_alias"],
        "oracle_token_f1": oracle_match["token_f1"],
        "oracle_token_set_ratio": oracle_match["token_set_ratio"],
        "oracle_correct_strict": oracle_match["token_f1"] >= strict_f1_cutoff,
    }


# -----------------------------
# 3. Single-pass experiments
# -----------------------------

LABEL_EXPERIMENTS = [
    {
        "name": "product_type",
        "labels": ["product type"],
        "target_labels": ["product type"],
    },
    {
        "name": "generic_product_type",
        "labels": ["generic product type"],
        "target_labels": ["generic product type"],
    },
    {
        "name": "main_product_object",
        "labels": ["main product object"],
        "target_labels": ["main product object"],
    },
    {
        "name": "physical_product_object",
        "labels": ["physical product object"],
        "target_labels": ["physical product object"],
    },
    {
        "name": "main_item_being_sold",
        "labels": ["main item being sold"],
        "target_labels": ["main item being sold"],
    },
    {
        "name": "generic_with_min_decoys",
        "labels": [
            "generic product type",
            "brand",
            "product line",
            "model number",
            "compatible product",
            "included accessory",
        ],
        "target_labels": ["generic product type"],
    },
    {
        "name": "main_object_with_min_decoys",
        "labels": [
            "main product object",
            "brand",
            "product line",
            "model number",
            "compatible product",
            "included accessory",
        ],
        "target_labels": ["main product object"],
    },
    {
        "name": "generic_with_rich_decoys",
        "labels": [
            "generic product type",
            "brand",
            "product line",
            "model number",
            "compatible product",
            "included accessory",
            "product attribute",
            "size",
            "application",
        ],
        "target_labels": ["generic product type"],
    },
]


def run_single_pass_experiment(
    model,
    gold: List[Dict[str, Any]],
    exp: Dict[str, Any],
    threshold: float,
    batch_size: int = 16,
    strict_f1_cutoff: float = 0.80,
    flat_ner: bool = True,
    multi_label: bool = False,
) -> pd.DataFrame:
    titles = [x["title"] for x in gold]

    predictions = model.inference(
        titles,
        exp["labels"],
        batch_size=batch_size,
        threshold=threshold,
        flat_ner=flat_ner,
        multi_label=multi_label,
    )

    rows = []
    for i, item in enumerate(gold):
        score = score_predictions_for_one_title(
            predictions[i],
            item["aliases"],
            target_labels=exp["target_labels"],
            strict_f1_cutoff=strict_f1_cutoff,
        )

        rows.append({
            "idx": i,
            "title": item["title"],
            "gold_aliases": item["aliases"],
            "experiment": exp["name"],
            "threshold": threshold,
            "labels": exp["labels"],
            "target_labels": exp["target_labels"],
            **score,
            "all_predictions": predictions[i],
        })

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["experiment", "threshold"], as_index=False)
        .agg(
            n=("idx", "count"),
            selected_acc_strict=("selected_correct_strict", "mean"),
            oracle_acc_strict=("oracle_correct_strict", "mean"),
            mean_selected_f1=("selected_token_f1", "mean"),
            mean_oracle_f1=("oracle_token_f1", "mean"),
            avg_target_preds=("n_target_preds", "mean"),
            no_target_pred_rate=("n_target_preds", lambda s: float((s == 0).mean())),
        )
        .sort_values(
            # ["selected_acc_strict", "mean_selected_f1", "oracle_acc_strict"],
            ["mean_selected_f1", "mean_oracle_f1"],
            ascending=False,
        )
    )


def run_single_pass_grid(
    model,
    gold: List[Dict[str, Any]],
    label_experiments: List[Dict[str, Any]] = LABEL_EXPERIMENTS,
    thresholds: List[float] = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    batch_size: int = 16,
    strict_f1_cutoff: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []

    for exp in tqdm(label_experiments, desc="label experiments"):
        for threshold in tqdm(thresholds, desc=f"{exp['name']} thresholds", leave=False):
            df = run_single_pass_experiment(
                model=model,
                gold=gold,
                exp=exp,
                threshold=threshold,
                batch_size=batch_size,
                strict_f1_cutoff=strict_f1_cutoff,
                flat_ner=True,
                multi_label=False,
            )
            all_rows.append(df)

    detailed = pd.concat(all_rows, ignore_index=True)
    summary = summarize_results(detailed)
    return detailed, summary


# -----------------------------
# 4. Optional two-pass GLiNER-only experiment
# -----------------------------

def run_two_pass_experiment(
    model,
    gold: List[Dict[str, Any]],
    pass1_labels: List[str],
    pass2_labels: List[str],
    pass2_target_labels: List[str],
    pass1_threshold: float,
    pass2_threshold: float,
    batch_size: int = 16,
    strict_f1_cutoff: float = 0.80,
    fallback_to_pass1: bool = False,
    experiment_name: str = "two_pass",
) -> pd.DataFrame:
    """
    Pass 1:
      title -> broad candidate spans

    Pass 2:
      candidate span -> more specific labels

    No rule-based postprocessing is applied.
    """

    titles = [x["title"] for x in gold]

    pass1_preds = model.inference(
        titles,
        pass1_labels,
        batch_size=batch_size,
        threshold=pass1_threshold,
        flat_ner=True,
        multi_label=False,
    )

    # Flatten pass-1 spans into a batch for pass 2
    pass2_inputs = []
    owners = []

    for i, preds in enumerate(pass1_preds):
        if preds is None:
            continue

        for p in preds:
            span = str(p.get("text", "")).strip()
            if span:
                pass2_inputs.append(span)
                owners.append((i, p))

    final_preds = [[] for _ in gold]

    if pass2_inputs:
        pass2_outputs = model.inference(
            pass2_inputs,
            pass2_labels,
            batch_size=batch_size,
            threshold=pass2_threshold,
            flat_ner=True,
            multi_label=False,
        )

        for (i, p1), p2_preds in zip(owners, pass2_outputs):
            if p2_preds is None:
                continue

            for p2 in p2_preds:
                final_preds[i].append({
                    "text": p2.get("text"),
                    "label": p2.get("label"),
                    # combined score is arbitrary but useful for ranking
                    "score": float(p1.get("score", 0.0)) * float(p2.get("score", 0.0)),
                    "pass1_text": p1.get("text"),
                    "pass1_label": p1.get("label"),
                    "pass1_score": float(p1.get("score", 0.0)),
                    "pass2_score": float(p2.get("score", 0.0)),
                })

    # Optional fallback: if pass 2 returns no product-type label, use top pass-1 span.
    # This is still model-only, but it is selection logic.
    target_for_scoring = list(pass2_target_labels)

    if fallback_to_pass1:
        fallback_label = "__pass1_fallback__"
        target_for_scoring.append(fallback_label)

        for i, preds in enumerate(pass1_preds):
            has_target = any(p.get("label") in pass2_target_labels for p in final_preds[i])
            if has_target:
                continue

            if preds:
                top_p1 = max(preds, key=lambda p: float(p.get("score", 0.0)))
                final_preds[i].append({
                    "text": top_p1.get("text"),
                    "label": fallback_label,
                    "score": float(top_p1.get("score", 0.0)),
                    "pass1_text": top_p1.get("text"),
                    "pass1_label": top_p1.get("label"),
                    "pass1_score": float(top_p1.get("score", 0.0)),
                    "pass2_score": None,
                })

    rows = []
    for i, item in enumerate(gold):
        score = score_predictions_for_one_title(
            final_preds[i],
            item["aliases"],
            target_labels=target_for_scoring,
            strict_f1_cutoff=strict_f1_cutoff,
        )

        rows.append({
            "idx": i,
            "title": item["title"],
            "gold_aliases": item["aliases"],
            "experiment": experiment_name,
            "pass1_threshold": pass1_threshold,
            "pass2_threshold": pass2_threshold,
            "threshold": f"{pass1_threshold}->{pass2_threshold}",
            "pass1_labels": pass1_labels,
            "pass2_labels": pass2_labels,
            "target_labels": target_for_scoring,
            **score,
            "all_predictions": final_preds[i],
            "pass1_predictions": pass1_preds[i],
        })

    return pd.DataFrame(rows)


def run_two_pass_grid(
    model,
    gold: List[Dict[str, Any]],
    batch_size: int = 16,
    strict_f1_cutoff: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = [
        {
            "experiment_name": "two_pass_main_item_to_generic",
            "pass1_labels": ["main item being sold"],
            "pass2_labels": [
                "generic product type",
                "brand",
                "product line",
                "model number",
                "product attribute",
                "compatible product",
                "included accessory",
            ],
            "pass2_target_labels": ["generic product type"],
            "fallback_to_pass1": False,
        },
        {
            "experiment_name": "two_pass_main_object_to_generic",
            "pass1_labels": ["main product object"],
            "pass2_labels": [
                "generic product type",
                "brand",
                "product line",
                "model number",
                "product attribute",
                "compatible product",
                "included accessory",
            ],
            "pass2_target_labels": ["generic product type"],
            "fallback_to_pass1": False,
        },
        {
            "experiment_name": "two_pass_main_item_to_generic_with_fallback",
            "pass1_labels": ["main item being sold"],
            "pass2_labels": [
                "generic product type",
                "brand",
                "product line",
                "model number",
                "product attribute",
                "compatible product",
                "included accessory",
            ],
            "pass2_target_labels": ["generic product type"],
            "fallback_to_pass1": True,
        },
    ]

    pass1_thresholds = [0.30, 0.40, 0.50]
    pass2_thresholds = [0.25, 0.35, 0.45, 0.55]

    all_rows = []

    for cfg in tqdm(configs, desc="two-pass configs"):
        for t1 in tqdm(pass1_thresholds, desc=cfg["experiment_name"], leave=False):
            for t2 in pass2_thresholds:
                df = run_two_pass_experiment(
                    model=model,
                    gold=gold,
                    pass1_labels=cfg["pass1_labels"],
                    pass2_labels=cfg["pass2_labels"],
                    pass2_target_labels=cfg["pass2_target_labels"],
                    pass1_threshold=t1,
                    pass2_threshold=t2,
                    batch_size=batch_size,
                    strict_f1_cutoff=strict_f1_cutoff,
                    fallback_to_pass1=cfg["fallback_to_pass1"],
                    experiment_name=cfg["experiment_name"],
                )
                all_rows.append(df)

    detailed = pd.concat(all_rows, ignore_index=True)
    summary = summarize_results(detailed)
    return detailed, summary

model = GLiNER.from_pretrained(
    args.model_path,
    map_location="cuda",
    load_tokenizer=True,
)

model.to(torch.float16)
model.compile()

_ = model.inference(
    ["Warmup title for compilation"],
    ["product type"],
    batch_size=1,
    threshold=0.3,
)

single_detailed, single_summary = run_single_pass_grid(
    model=model,
    gold=GOLD,
    thresholds=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    batch_size=1,
    strict_f1_cutoff=0.80,
)

single_summary

print(single_summary.head(20))

two_detailed, two_summary = run_two_pass_grid(
    model=model,
    gold=GOLD,
    batch_size=1,
    strict_f1_cutoff=0.80,
)

print(two_summary.head(20))