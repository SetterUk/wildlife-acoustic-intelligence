"""
Taxonomy and Species Candidate Management for AnimalCLAP.

AnimalCLAP was pretrained with taxonomy-aware language representations,
incorporating Class, Order, Family, Genus, and Species metadata.
"""

from typing import List, Dict, Any, Optional

DEFAULT_CANDIDATE_SPECIES: List[Dict[str, str]] = [
    # --- Mammals ---
    {
        "category": "Mammals",
        "common_name": "Bengal tiger",
        "scientific_name": "Panthera tigris",
        "class": "Mammalia",
        "order": "Carnivora",
        "family": "Felidae",
        "genus": "Panthera",
        "habitat": "Tropical and subtropical moist broadleaf forests",
        "vocalization": "Roar, chuff, growl"
    },
    {
        "category": "Mammals",
        "common_name": "Asian elephant",
        "scientific_name": "Elephas maximus",
        "class": "Mammalia",
        "order": "Proboscidea",
        "family": "Elephantidae",
        "genus": "Elephas",
        "habitat": "Grasslands, tropical evergreen and semi-evergreen forests",
        "vocalization": "Trumpet, rumble, roar, bark"
    },
    {
        "category": "Mammals",
        "common_name": "Indian leopard",
        "scientific_name": "Panthera pardus",
        "class": "Mammalia",
        "order": "Carnivora",
        "family": "Felidae",
        "genus": "Panthera",
        "habitat": "Deciduous forests, scrub jungles, rock outcrops",
        "vocalization": "Sawing cough, growl, purr"
    },
    {
        "category": "Mammals",
        "common_name": "Indian grey wolf",
        "scientific_name": "Canis lupus",
        "class": "Mammalia",
        "order": "Carnivora",
        "family": "Canidae",
        "genus": "Canis",
        "habitat": "Open scrublands, dry grasslands, arid thorn forests",
        "vocalization": "Harmonic pack howl, whimper, bark"
    },
    {
        "category": "Mammals",
        "common_name": "Spotted deer",
        "scientific_name": "Axis axis",
        "class": "Mammalia",
        "order": "Artiodactyla",
        "family": "Cervidae",
        "genus": "Axis",
        "habitat": "Open grasslands, moist and dry deciduous forests",
        "vocalization": "High-pitched alarm bark, distress bleat"
    },
    {
        "category": "Mammals",
        "common_name": "Common langur",
        "scientific_name": "Semnopithecus entellus",
        "class": "Mammalia",
        "order": "Primates",
        "family": "Cercopithecidae",
        "genus": "Semnopithecus",
        "habitat": "Forest canopies, agricultural areas, human settlements",
        "vocalization": "Whoop call, guttural alarm alarm bark, hiccup"
    },
    {
        "category": "Mammals",
        "common_name": "Indian rhinoceros",
        "scientific_name": "Rhinoceros unicornis",
        "class": "Mammalia",
        "order": "Perissodactyla",
        "family": "Rhinocerotidae",
        "genus": "Rhinoceros",
        "habitat": "Alluvial grasslands and riverine swamp forests",
        "vocalization": "Snort, grunt, honk, bleat, roar"
    },
    {
        "category": "Mammals",
        "common_name": "White rhinoceros",
        "scientific_name": "Ceratotherium simum",
        "class": "Mammalia",
        "order": "Perissodactyla",
        "family": "Rhinocerotidae",
        "genus": "Ceratotherium",
        "habitat": "Grasslands and open acacia savannas",
        "vocalization": "Resonant snort, pant, squeak, growl"
    },

    # --- Birds ---
    {
        "category": "Birds",
        "common_name": "Indian peafowl",
        "scientific_name": "Pavo cristatus",
        "class": "Aves",
        "order": "Galliformes",
        "family": "Phasianidae",
        "genus": "Pavo",
        "habitat": "Deciduous forests, agricultural margins, riparian scrub",
        "vocalization": "Resonant trumpeting 'may-awe' call, alarm honk"
    },
    {
        "category": "Birds",
        "common_name": "Common myna",
        "scientific_name": "Acridotheres tristis",
        "class": "Aves",
        "order": "Passeriformes",
        "family": "Sturnidae",
        "genus": "Acridotheres",
        "habitat": "Woodlands, cultivation, urban and suburban areas",
        "vocalization": "Complex chatter, high whistles, scolding squawks"
    },
    {
        "category": "Birds",
        "common_name": "Indian crow",
        "scientific_name": "Corvus splendens",
        "class": "Aves",
        "order": "Passeriformes",
        "family": "Corvidae",
        "genus": "Corvus",
        "habitat": "Urban centers, open country, coastal villages",
        "vocalization": "Harsh nasal 'kaaa-kaaa' caw"
    },
    {
        "category": "Birds",
        "common_name": "House sparrow",
        "scientific_name": "Passer domesticus",
        "class": "Aves",
        "order": "Passeriformes",
        "family": "Passeridae",
        "genus": "Passer",
        "habitat": "Human habitations, rural villages, farmland",
        "vocalization": "Short rhythmic 'chirrup' and chattering social calls"
    },
    {
        "category": "Birds",
        "common_name": "Indian koel",
        "scientific_name": "Eudynamys scolopaceus",
        "class": "Aves",
        "order": "Cuculiformes",
        "family": "Cuculidae",
        "genus": "Eudynamys",
        "habitat": "Dense canopy woodlands, botanical gardens, orchards",
        "vocalization": "Rising crescendo 'ku-oo ku-oo' call"
    },
    {
        "category": "Birds",
        "common_name": "Indian eagle-owl",
        "scientific_name": "Bubo bengalensis",
        "class": "Aves",
        "order": "Strigiformes",
        "family": "Strigidae",
        "genus": "Bubo",
        "habitat": "Rocky hillsides, bush-covered ravines, ruins",
        "vocalization": "Deep resonant double hoot 'tu-whoo'"
    },

    # --- Amphibians ---
    {
        "category": "Amphibians",
        "common_name": "Indian bullfrog",
        "scientific_name": "Hoplobatrachus tigerinus",
        "class": "Amphibia",
        "order": "Anura",
        "family": "Dicroglossidae",
        "genus": "Hoplobatrachus",
        "habitat": "Wetlands, paddy fields, temporary monsoon pools",
        "vocalization": "Deep guttural bellowing quack or grunt"
    },
    {
        "category": "Amphibians",
        "common_name": "Common tree frog",
        "scientific_name": "Polypedates maculatus",
        "class": "Amphibia",
        "order": "Anura",
        "family": "Rhacophoridae",
        "genus": "Polypedates",
        "habitat": "Moist deciduous forests, gardens, near human dwellings",
        "vocalization": "Series of hollow clicking and drumming taps"
    },
    {
        "category": "Amphibians",
        "common_name": "Midwife toad / Tree frog",
        "scientific_name": "Alytes obstetricans",
        "class": "Amphibia",
        "order": "Anura",
        "family": "Alytidae",
        "genus": "Alytes",
        "habitat": "Riparian forests, water pools, rocky banks",
        "vocalization": "Clear melodic electronic-piping bell tone"
    },

    # --- Reptiles ---
    {
        "category": "Reptiles",
        "common_name": "Mugger crocodile",
        "scientific_name": "Crocodylus palustris",
        "class": "Reptilia",
        "order": "Crocodilia",
        "family": "Crocodylidae",
        "genus": "Crocodylus",
        "habitat": "Freshwater rivers, lakes, marshes, reservoirs",
        "vocalization": "Low-frequency sub-audible head slap rumble, hatchling yelp"
    },
    {
        "category": "Reptiles",
        "common_name": "Tokay gecko",
        "scientific_name": "Gekko gecko",
        "class": "Reptilia",
        "order": "Squamata",
        "family": "Gekkonidae",
        "genus": "Gekko",
        "habitat": "Tropical rainforest trees and rustic wooden dwellings",
        "vocalization": "Loud two-phase 'to-kay' barking vocalization"
    },

    # --- Insects ---
    {
        "category": "Insects",
        "common_name": "House cricket",
        "scientific_name": "Acheta domesticus",
        "class": "Insecta",
        "order": "Orthoptera",
        "family": "Gryllidae",
        "genus": "Acheta",
        "habitat": "Fields, caves, under rocks, indoor environments",
        "vocalization": "High-frequency wing stridulation trill"
    },
    {
        "category": "Insects",
        "common_name": "Periodical cicada",
        "scientific_name": "Magicicada septendecim",
        "class": "Insecta",
        "order": "Hemiptera",
        "family": "Cicadidae",
        "genus": "Magicicada",
        "habitat": "Deciduous hardwood canopies and forests",
        "vocalization": "Continuous deafening tymbal vibration drone"
    },

    # --- Marine Animals ---
    {
        "category": "Marine animals",
        "common_name": "Dolphin",
        "scientific_name": "Tursiops truncatus",
        "class": "Mammalia",
        "order": "Artiodactyla",
        "family": "Delphinidae",
        "genus": "Tursiops",
        "habitat": "Coastal waters, bays, open tropical and temperate oceans",
        "vocalization": "Broadband echolocation clicks, signature frequency-modulated whistle"
    },
    {
        "category": "Marine animals",
        "common_name": "Humpback whale",
        "scientific_name": "Megaptera novaeangliae",
        "class": "Mammalia",
        "order": "Artiodactyla",
        "family": "Balaenopteridae",
        "genus": "Megaptera",
        "habitat": "Pelagic oceans and polar-to-tropical migratory corridors",
        "vocalization": "Complex melodic low-frequency songs, moans, and groans"
    }
]

def format_taxonomy_prompt(species: Dict[str, str], style: str = "tax_com") -> str:
    """
    Format a taxonomy-aware prompt matching AnimalCLAP pretraining text modes.
    
    Styles:
        - 'tax_com': '{tax} with common name {com}' (Default, recommended by AnimalCLAP)
        - 'sci_com': '{sci} with common name {com}'
        - 'common': '{com}'
        - 'scientific': '{sci}'
        - 'hierarchical': '{class} {order} {family} {genus} {sci}'
    """
    com = species.get("common_name", "").strip()
    sci = species.get("scientific_name", "").strip()
    cls_ = species.get("class", "").strip()
    ord_ = species.get("order", "").strip()
    fam = species.get("family", "").strip()
    gen = species.get("genus", "").strip()

    tax = " ".join([x for x in [cls_, ord_, fam, gen, sci] if x])

    if style == "tax_com":
        if tax and com:
            return f"{tax} with common name {com}"
        return com or sci or "animal"
    elif style == "sci_com":
        if sci and com:
            return f"{sci} with common name {com}"
        return com or sci
    elif style == "scientific":
        return sci or com
    elif style == "hierarchical":
        return tax or com
    else:
        return com or sci

