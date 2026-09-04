"""Seed common ingredient aliases into the reference database (idempotent).

DDInter uses international names (e.g. 'Acetaminophen'); patients in Yemen often
say 'paracetamol'. The Ingredient.aliases column is exactly the place for such
synonyms — this script adds them so both the text check and the scan pipeline
resolve them. Extend ALIASES freely; existing aliases are never duplicated.

Usage: python scripts/seed_aliases.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG  # noqa: E402
from db.database import Database  # noqa: E402
from db.repositories import IngredientRepository  # noqa: E402

# (alias, canonical ingredient name in the DB) — canonical names follow DDInter.
ALIASES: list[tuple[str, str]] = [
    # Aspirin
    ("aspirin", "acetylsalicylic acid"),
    ("asprin", "acetylsalicylic acid"),
    ("asa", "acetylsalicylic acid"),
    ("bayer aspirin", "acetylsalicylic acid"),
    ("ecotrin", "acetylsalicylic acid"),
    ("aspocid", "acetylsalicylic acid"),
    ("baby aspirin", "acetylsalicylic acid"),
    # Paracetamol / Acetaminophen
    ("paracetamol", "acetaminophen"),
    ("acetaminofen", "acetaminophen"),
    ("panadol", "acetaminophen"),
    ("tylenol", "acetaminophen"),
    ("calpol", "acetaminophen"),
    ("adadol", "acetaminophen"),
    ("fevadol", "acetaminophen"),
    # NSAIDs
    ("advil", "ibuprofen"),
    ("brufen", "ibuprofen"),
    ("motrin", "ibuprofen"),
    ("profin", "ibuprofen"),
    ("voltaren", "diclofenac"),
    ("cataflam", "diclofenac"),
    ("olphen", "diclofenac"),
    ("ponstan", "mefenamic acid"),
    ("aleve", "naproxen"),
    ("naprosyn", "naproxen"),
    ("feldene", "piroxicam"),
    ("mobic", "meloxicam"),
    ("celebrex", "celecoxib"),
    # Antibiotics
    ("amoxil", "amoxicillin"),
    ("augmentin", "amoxicillin"),
    ("curam", "amoxicillin"),
    ("klacid", "clarithromycin"),
    ("zithromax", "azithromycin"),
    ("cipro", "ciprofloxacin"),
    ("ciprobay", "ciprofloxacin"),
    ("tavanic", "levofloxacin"),
    ("flagyl", "metronidazole"),
    ("bactrim", "sulfamethoxazole"),
    ("septrin", "sulfamethoxazole"),
    ("vibramycin", "doxycycline"),
    # Cardiovascular & Blood
    ("plavix", "clopidogrel"),
    ("coumadin", "warfarin"),
    ("marevan", "warfarin"),
    ("eliquis", "apixaban"),
    ("xarelto", "rivaroxaban"),
    ("lipitor", "atorvastatin"),
    ("crestor", "rosuvastatin"),
    ("zocor", "simvastatin"),
    ("concor", "bisoprolol"),
    ("inderal", "propranolol"),
    ("tenormin", "atenolol"),
    ("norvasc", "amlodipine"),
    ("lasix", "furosemide"),
    ("aldactone", "spironolactone"),
    ("capoten", "captopril"),
    ("zestril", "lisinopril"),
    ("cozaar", "losartan"),
    ("diovan", "valsartan"),
    ("lanoxin", "digoxin"),
    # Diabetes & GI & Others
    ("glucophage", "metformin"),
    ("amaryl", "glimepiride"),
    ("daonil", "glibenclamide"),
    ("januvia", "sitagliptin"),
    ("nexium", "esomeprazole"),
    ("losec", "omeprazole"),
    ("pantozol", "pantoprazole"),
    ("controloc", "pantoprazole"),
    ("zantac", "ranitidine"),
    ("motilium", "domperidone"),
    ("plasil", "metoclopramide"),
    ("ventolin", "albuterol"),
]


def main() -> int:
    database = Database(CONFIG.database)
    added_aliases = 0
    added_drugs = 0
    with database.session() as session:
        from db.repositories import DrugRepository

        ingredients = IngredientRepository(session)
        drugs = DrugRepository(session)

        for alias, canonical in ALIASES:
            ingredient = ingredients.get_by_name(canonical)
            if ingredient is None:
                print(f"skip: canonical '{canonical}' not in DB")
                continue
            aliases = json.loads(ingredient.aliases or "[]")
            if alias not in aliases:
                aliases.append(alias)
                ingredient.aliases = json.dumps(aliases, ensure_ascii=False)
                added_aliases += 1
                print(f"aliased: {alias} -> {canonical}")

            # Also ensure a friendly capitalized Drug entry exists for pickers
            drug_name = alias.capitalize()
            if drugs.get_by_name(drug_name) is None and drugs.get_by_name(alias) is None:
                drugs.add(trade_name=drug_name, ingredient_ids=[ingredient.id])
                added_drugs += 1
                print(f"added drug: {drug_name} (active: {canonical})")

        session.commit()
    print(f"done: {added_aliases} alias(es) added, {added_drugs} new drug(s) added")
    return 0


if __name__ == "__main__":
    sys.exit(main())
