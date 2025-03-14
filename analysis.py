import argparse
import difflib
import json
import os
import sys
import uuid

from datetime import datetime as dt
from enum import Enum
from pathlib import Path

from googleapiclient.discovery import build
from google.oauth2 import service_account
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from pydantic import BaseModel


SERVICE_ACCOUNT_FILE = "./sa.json"
PROJECT_ID = os.getenv("PROJECT_ID") or sys.exit("PROJECT_ID is not set")
LOCATION = os.getenv("LOCATION") or "us-central1"
SAVE_EXPERIMENT_DETAILS = os.getenv("SAVE_EXPERIMENT_DETAILS") or False


class SuggestionType(Enum):
    DEFAULT_FOR_CURRENT_ACCESS = "DEFAULT"
    SUGGESTIONS_INLINE = "SUGGESTIONS_INLINE"
    PREVIEW_SUGGESTIONS_ACCEPTED = "PREVIEW_SUGGESTIONS_ACCEPTED"
    PREVIEW_WITHOUT_SUGGESTIONS = "PREVIEW_WITHOUT_SUGGESTIONS"


class Impact(Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class ChangeSummary(BaseModel):
    body: str
    section_name: str
    document_type: str
    contractor_name: str
    impact: Impact


client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1"))


def get_document(document_id, suggestions_view_mode) -> dict:
    SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    docs_service = build("docs", "v1", credentials=credentials)

    document = docs_service.documents().get(
      documentId=document_id,
      includeTabsContent=True,
      suggestionsViewMode=suggestions_view_mode
    ).execute()

    # print("The title of the document is: {}".format(document.get("title")))

    return document


def full_content(doc) -> str:
    text_content = []

    tab = doc.get("tabs")[0].get("documentTab")

    for struct in tab.get("body").get("content"):
        if struct.get("paragraph") is not None:
            paragraph = struct.get("paragraph")

            for element in paragraph.get("elements"):
                if element.get("textRun") is not None:
                    text_run = element.get("textRun")
                    if text_run.get("content") is not None:
                        text_content.append(text_run.get("content"))

    return "".join(text_content)


def document_diff(base: str, modified: str) -> str:
    diff = difflib.unified_diff(
        base.splitlines(),
        modified.splitlines(),
        fromfile="base",
        tofile="modified",
    )

    return "\n".join(diff)


def get_prompt(contract_diff: str, contract_modified: str) -> str:
    prompt = f"""
Review and summarize the changes suggested to our base contract by a 3rd-party vendor. This should be a summary, not a citation.
The shorter and more accurate each summary - the better. Filler words and empty chatter in response are prohibited.

Analyze each proposed contract change and classify its qualitative and quantitative impact.
Use the full body of the modified document if you need more context to produce an accurate summary or impact analysis.

Finally, guess the document type and the contractor name from this contract document.
Generally used document types are:

* ISA (Inbound Services Agreement)
* SOW (Statement of work) - basically a “Contract” listing out details of the work, engagement, terms etc
* SDW (Scope Discussion Worksheet)

# Diff

{contract_diff}

# Modified contract

{contract_modified}
"""

    return prompt


def save_experiment_artifacts(
    document_id: str,
    base: str,
    modified: str,
    diff: str,
    prompt: str,
    settings: dict,
    response: list[str],
    metadata: dict,
):
    experiment_id, t = uuid.uuid4().hex[:10], dt.now().strftime("%m-%d_T%H-%M-%S")
    print(f"Experiment ID & Time: {experiment_id} / {t}")
    output_dir = Path(f"./tmp/{document_id}/{t}_{experiment_id}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(f"./{output_dir}/base.txt", "w") as f:
        f.write(base)

    with open(f"./{output_dir}/modified.txt", "w") as f:
        f.write(modified)

    with open(f"./{output_dir}/diff.txt", "w") as f:
        f.write(diff)

    with open(f"./{output_dir}/prompt.txt", "w") as f:
        f.write(prompt)

    with open(f"./{output_dir}/settings.json", "w") as f:
        f.write(json.dumps(settings, indent=2))

    with open(f"./{output_dir}/response.json", "w") as f:
        f.write(json.dumps(response, indent=2))

    with open(f"./{output_dir}/metadata.json", "w") as f:
        f.write(str(metadata))

    return experiment_id, output_dir


def ask_gemini(model: str, prompt: str, seed: int, temperature: float, top_p: float):
    # print("Asking Gemini:\n{}".format(prompt))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=GenerateContentConfig(
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            response_mime_type="application/json",
            response_schema=list[ChangeSummary],
        ),
    )

    return response


def cli():
    parser = argparse.ArgumentParser(description="Analyze changes in a contract document")

    parser.add_argument("-id", "--document-id", type=str, required=True, help="Contract ID")

    parser.add_argument("--model", type=str, default="gemini-1.5-pro-002", help="Gemini model to use")
    parser.add_argument("--seed", type=int, default=10042, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top P")

    return parser.parse_args()


def unpack_gemini_response(response):
    # print("Usage metadata:\n", response.usage_metadata)
    # print("Gemini says:\n", response.text)
    # print("Parsed:")

    items = [item.model_dump(mode="json") for item in response.parsed]
    metadata = response.usage_metadata

    return items, metadata


def run_analysis(document_id: str, model: str, seed: int, temperature: float, top_p: float):
    full_document_base = full_content(
        get_document(document_id, SuggestionType.PREVIEW_WITHOUT_SUGGESTIONS.value))
    full_document_modified = full_content(
        get_document(document_id, SuggestionType.PREVIEW_SUGGESTIONS_ACCEPTED.value))

    diff = document_diff(full_document_base, full_document_modified)

    prompt = get_prompt(diff, full_document_modified)

    settings = {
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = ask_gemini(prompt=prompt, **settings)

    items, metadata = unpack_gemini_response(response)

    if SAVE_EXPERIMENT_DETAILS:
        _, output_dir = save_experiment_artifacts(
            document_id=document_id,
            base=full_document_base,
            modified=full_document_modified,
            diff=diff,
            prompt=prompt,
            settings=settings,
            response=items,
            metadata=metadata,
        )
        print(f"Experiment artifacts saved to {output_dir}")

    return items, metadata

def main():
    args = cli()

    items, metadata = run_analysis(
        document_id=args.document_id,
        model=args.model,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("Usage metadata:\n", metadata)
    print(json.dumps(items, indent=2))


def test():
    args = cli()

    full_document_base = full_content(
        get_document(args.document_id, SuggestionType.PREVIEW_WITHOUT_SUGGESTIONS.value))

    prompt = f"""
Guess the document type and the contractor name from this contract document. Note, that "Google" is NOT the contractor, - it's the client.
Generally used document types are:

* ISA (Inbound Services Agreement)
* SOW (Statement of work) - basically a “Contract” listing out details of the work, engagement, terms etc
* SDW (Scope Discussion Worksheet)

# Document body

{full_document_base}
"""

    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=GenerateContentConfig(
            # seed=seed,
            # temperature=temperature,
            # top_p=top_p,
            response_mime_type="application/json",
            # response_schema=list[ChangeSummary],
        ),
    )

    print(response.text)

if __name__ == "__main__":
    main()
    # test()
