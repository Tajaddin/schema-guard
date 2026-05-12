"""Adversarial structured-output benchmark dataset.

Each task pairs a Pydantic schema with an extraction prompt. The prompts are
crafted to elicit common malformed-JSON failure modes that I have actually seen
in production:

* trailing commas
* markdown fence wrapping
* smart quotes
* prose mixed with JSON
* numeric values formatted as text ("$1,000" vs 1000)
* additional fields the schema does not allow
* missing required fields
* enum values with wrong case

The intent is NOT that any individual LLM call must fail — modern Haiku / 4o-mini
get most of these right on a first try. The intent is that across 60 prompts, the
4% of calls that go wrong are the ones every production system has to handle.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Person(BaseModel):
    name: str
    age: int


class Address(BaseModel):
    street: str
    city: str
    country: str


class PersonWithAddress(BaseModel):
    name: str
    address: Address


class Product(BaseModel):
    sku: str
    price_usd: float
    in_stock: bool


class ProductList(BaseModel):
    products: list[Product]


class Event(BaseModel):
    title: str
    date: str  # ISO format
    location: str | None = None


class Ticket(BaseModel):
    id: str
    severity: Literal["low", "medium", "high", "critical"]
    summary: str


class TicketList(BaseModel):
    tickets: list[Ticket] = Field(default_factory=list)


class Recipe(BaseModel):
    name: str
    servings: int
    ingredients: list[str]
    time_minutes: int


class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    headquarters: str
    revenue_usd_billions: float | None = None


class WeatherForecast(BaseModel):
    location: str
    temperature_f: float
    conditions: str


# ---------------------------------------------------------------------------
# Tasks: (id, prompt, schema, gold_answer_substring)
# ---------------------------------------------------------------------------

TASKS: list[dict] = [
    # --- Person extraction (10 cases) ---
    {
        "id": "person-basic-1",
        "prompt": "Extract the person: Alice Johnson is 34 years old.",
        "schema": Person,
        "expected_name": "Alice Johnson",
    },
    {
        "id": "person-basic-2",
        "prompt": "Person info from: Bob Smith, age 55.",
        "schema": Person,
        "expected_name": "Bob Smith",
    },
    {
        "id": "person-prose-1",
        "prompt": "Tell me about Carol Davis who turned 27 last month. Give me JSON with name and age fields.",
        "schema": Person,
        "expected_name": "Carol Davis",
    },
    {
        "id": "person-prose-2",
        "prompt": "Please reply with a JSON object describing the person named Dan Lee (age 19).",
        "schema": Person,
        "expected_name": "Dan Lee",
    },
    {
        "id": "person-tricky-quotes",
        "prompt": "Extract person from text: \"Eve O'Brien\" age 41. Output JSON.",
        "schema": Person,
        "expected_name": "Eve",  # quote-sensitive
    },
    {
        "id": "person-format-fight",
        "prompt": "JSON only please. Extract: Frank García-Mendez, 67 years.",
        "schema": Person,
        "expected_name": "Frank",
    },
    {
        "id": "person-number-as-word",
        "prompt": "Extract person from: Grace is thirty-two years old. Return JSON.",
        "schema": Person,
        "expected_name": "Grace",
    },
    {
        "id": "person-name-only",
        "prompt": "Henry. Make a JSON object with at least the name field.",
        "schema": Person,  # age required → forces tier escalation
        "expected_name": "Henry",
    },
    {
        "id": "person-multiline",
        "prompt": "Extract person:\n\nName: Isabel Rojas\nAge: 38\n\nReturn JSON.",
        "schema": Person,
        "expected_name": "Isabel",
    },
    {
        "id": "person-comma-numbers",
        "prompt": "Extract: Jack King, age 1,001 (long-lived character). Return JSON.",
        "schema": Person,
        "expected_name": "Jack",
    },
    # --- Address / nested (8 cases) ---
    {
        "id": "address-basic",
        "prompt": "Extract person with address: Kate Lin lives at 100 Main Street, Boston, USA.",
        "schema": PersonWithAddress,
        "expected_name": "Kate",
    },
    {
        "id": "address-international",
        "prompt": "Extract person: Liam Brown, 12 Rue de la Paix, Paris, France.",
        "schema": PersonWithAddress,
        "expected_name": "Liam",
    },
    {
        "id": "address-no-country",
        "prompt": "Person at 5 Acacia Avenue, Manchester: Mia Patel. Country is United Kingdom.",
        "schema": PersonWithAddress,
        "expected_name": "Mia",
    },
    {
        "id": "address-comma-in-street",
        "prompt": "Extract: Noah Kim, Apt 4B, 7 Oak Drive, Seattle, USA.",
        "schema": PersonWithAddress,
        "expected_name": "Noah",
    },
    {
        "id": "address-tight",
        "prompt": "JSON: Olivia Park / 30 Sunset Blvd / LA / USA.",
        "schema": PersonWithAddress,
        "expected_name": "Olivia",
    },
    {
        "id": "address-prose-heavy",
        "prompt": "Please extract address details from this text and respond with JSON only: 'Pedro Sanchez lives at Carrer Mallorca 123 in Barcelona, Spain.'",
        "schema": PersonWithAddress,
        "expected_name": "Pedro",
    },
    {
        "id": "address-missing-fields",
        "prompt": "Extract person and address: Quinn from 22 Bay St, Toronto. (No country given — assume Canada.)",
        "schema": PersonWithAddress,
        "expected_name": "Quinn",
    },
    {
        "id": "address-non-latin",
        "prompt": "Extract person: 田中 太郎 (Tanaka Taro) lives at 1-1 Shibuya, Tokyo, Japan.",
        "schema": PersonWithAddress,
        "expected_name": "Tanaka",
    },
    # --- Products / arrays (8 cases) ---
    {
        "id": "products-basic",
        "prompt": "List the SKU-1234 widget at $19.99 in stock. JSON product list.",
        "schema": ProductList,
        "expected_name": "SKU-1234",
    },
    {
        "id": "products-multi",
        "prompt": "Inventory: SKU-A1 $5.00 in-stock, SKU-B2 $12.50 out-of-stock, SKU-C3 $99 in-stock. JSON.",
        "schema": ProductList,
        "expected_name": "SKU-A1",
    },
    {
        "id": "products-comma-prices",
        "prompt": "Items: SKU-RUG $1,499.00 in stock, SKU-LAMP $299.99 in stock. JSON product list.",
        "schema": ProductList,
        "expected_name": "SKU-RUG",
    },
    {
        "id": "products-mixed-currency",
        "prompt": "Item SKU-TEA at €4.99 (convert: ~$5.40), in stock. JSON.",
        "schema": ProductList,
        "expected_name": "SKU-TEA",
    },
    {
        "id": "products-empty-allowed",
        "prompt": "No products available today. JSON product list (empty is fine).",
        "schema": ProductList,
        "expected_name": "",
    },
    {
        "id": "products-prose",
        "prompt": "We have SKU-Q-1 (quill) for $0.50, ten in stock; SKU-Q-2 (ink) for $1.20, none in stock. Return product JSON.",
        "schema": ProductList,
        "expected_name": "SKU-Q-1",
    },
    {
        "id": "products-very-many",
        "prompt": "List products: SKU-1 $1, SKU-2 $2, SKU-3 $3, SKU-4 $4, SKU-5 $5. All in stock. JSON.",
        "schema": ProductList,
        "expected_name": "SKU-1",
    },
    {
        "id": "products-decimal-comma",
        "prompt": "European format: SKU-EU at 19,99 EUR in stock. JSON product list with USD-formatted price.",
        "schema": ProductList,
        "expected_name": "SKU-EU",
    },
    # --- Tickets / enum cases (6 cases) ---
    {
        "id": "ticket-clean",
        "prompt": "Create ticket TICKET-001 severity: high — summary: API timeout on /v2/users. JSON.",
        "schema": Ticket,
        "expected_name": "TICKET-001",
    },
    {
        "id": "ticket-wrong-case",
        "prompt": "Open ticket TICKET-2 with severity HIGH about login failures. JSON.",
        "schema": Ticket,
        "expected_name": "TICKET-2",
    },
    {
        "id": "ticket-paraphrase-severity",
        "prompt": "Critical incident TICKET-77: production database is down. JSON.",
        "schema": Ticket,
        "expected_name": "TICKET-77",
    },
    {
        "id": "ticket-list-empty",
        "prompt": "No tickets to report today. JSON ticket list (empty array fine).",
        "schema": TicketList,
        "expected_name": "",
    },
    {
        "id": "ticket-list-multi",
        "prompt": "Tickets: T-1 low: ui glitch; T-2 medium: slow page; T-3 critical: outage. JSON list.",
        "schema": TicketList,
        "expected_name": "T-1",
    },
    {
        "id": "ticket-severity-prose",
        "prompt": "TICKET-MEH is a minor issue (severity is low) about the icon color. JSON.",
        "schema": Ticket,
        "expected_name": "TICKET-MEH",
    },
    # --- Recipe (5 cases) ---
    {
        "id": "recipe-basic",
        "prompt": "Recipe: Pancakes. 4 servings. Flour, milk, eggs, butter. 20 minutes. JSON.",
        "schema": Recipe,
        "expected_name": "Pancakes",
    },
    {
        "id": "recipe-ingredients-prose",
        "prompt": "Quick Curry — 2 servings — needs onions, garlic, ginger, and tomatoes. About 30 min. JSON.",
        "schema": Recipe,
        "expected_name": "Curry",
    },
    {
        "id": "recipe-comma-ingredients",
        "prompt": "Caesar salad, 4 servings, 15 minutes. Ingredients: lettuce, parmesan, croutons, anchovies, dressing. JSON.",
        "schema": Recipe,
        "expected_name": "Caesar",
    },
    {
        "id": "recipe-mixed-units",
        "prompt": "Sourdough bread. 12 servings. Flour, water, salt, starter. Takes 24 hours from start to finish (1440 minutes). JSON.",
        "schema": Recipe,
        "expected_name": "Sourdough",
    },
    {
        "id": "recipe-extra-fields-attempt",
        "prompt": "Pasta Carbonara recipe (cuisine: Italian, difficulty: easy). 4 servings, 25 minutes. Spaghetti, eggs, pancetta, pecorino, pepper. JSON — only the canonical fields.",
        "schema": Recipe,
        "expected_name": "Carbonara",
    },
    # --- Company info (5 cases) ---
    {
        "id": "company-basic",
        "prompt": "Acme Corp founded 1942 in Burbank, CA. JSON company info.",
        "schema": CompanyInfo,
        "expected_name": "Acme",
    },
    {
        "id": "company-with-revenue",
        "prompt": "Globex was founded in 1989, HQ in Springfield. Revenue ~$2.3B last year. JSON.",
        "schema": CompanyInfo,
        "expected_name": "Globex",
    },
    {
        "id": "company-no-revenue",
        "prompt": "Initech, founded 1992 in Austin, TX. Revenue not disclosed. JSON.",
        "schema": CompanyInfo,
        "expected_name": "Initech",
    },
    {
        "id": "company-prose-heavy",
        "prompt": "The company known as Soylent Corp was originally founded in the year 1973 with HQ in New York. JSON only.",
        "schema": CompanyInfo,
        "expected_name": "Soylent",
    },
    {
        "id": "company-decimal-revenue",
        "prompt": "Yoyodyne Propulsion, est. 1955, HQ Grover's Mill NJ. Revenue $0.45B. JSON.",
        "schema": CompanyInfo,
        "expected_name": "Yoyodyne",
    },
    # --- Weather (5 cases) ---
    {
        "id": "weather-fahrenheit",
        "prompt": "Boston: 72F sunny. JSON weather forecast.",
        "schema": WeatherForecast,
        "expected_name": "Boston",
    },
    {
        "id": "weather-celsius-given",
        "prompt": "Paris: 22°C partly cloudy. JSON forecast with temperature in Fahrenheit.",
        "schema": WeatherForecast,
        "expected_name": "Paris",
    },
    {
        "id": "weather-prose",
        "prompt": "Tokyo right now: warm at about 81°F with light rain. Reply JSON.",
        "schema": WeatherForecast,
        "expected_name": "Tokyo",
    },
    {
        "id": "weather-negative-temp",
        "prompt": "Anchorage: -5°F clear and cold. JSON.",
        "schema": WeatherForecast,
        "expected_name": "Anchorage",
    },
    {
        "id": "weather-unicode-degree",
        "prompt": "Cairo: 95°F, dust. JSON.",
        "schema": WeatherForecast,
        "expected_name": "Cairo",
    },
    # --- Events (3 cases) ---
    {
        "id": "event-basic",
        "prompt": "Event: Anthropic Dev Day on 2026-05-15 at Pier 27, San Francisco. JSON.",
        "schema": Event,
        "expected_name": "Anthropic Dev Day",
    },
    {
        "id": "event-no-location",
        "prompt": "Event: Quarterly all-hands on 2026-06-01. Remote-only, no physical location. JSON.",
        "schema": Event,
        "expected_name": "all-hands",
    },
    {
        "id": "event-prose",
        "prompt": "The Spring Symposium happens 2026-04-22 at the University Auditorium. Return JSON.",
        "schema": Event,
        "expected_name": "Spring Symposium",
    },
]


def get_tasks(limit: int | None = None, seed: int | None = None) -> list[dict]:
    """Return the (optionally subsampled) task list."""
    import random

    tasks = list(TASKS)
    if seed is not None:
        random.Random(seed).shuffle(tasks)
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


SCHEMAS = {
    "Person": Person,
    "Address": Address,
    "PersonWithAddress": PersonWithAddress,
    "Product": Product,
    "ProductList": ProductList,
    "Event": Event,
    "Ticket": Ticket,
    "TicketList": TicketList,
    "Recipe": Recipe,
    "CompanyInfo": CompanyInfo,
    "WeatherForecast": WeatherForecast,
}
