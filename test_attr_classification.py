"""
Classify a preset paragraph against the closed vocabulary and print label scores.

Run: uv run python test_attr_classification.py
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from text_attributes import (
    SCORE_CUTOFF,
    VOCABULARY,
    active_labels_from_attributes,
    classify_text,
    derive_attributes,
    get_scorer,
)

for _name in ("httpx", "httpcore", "huggingface_hub", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

SAMPLE_PARAGRAPH = (
    '''NFL Madden is a popular American football video game series developed by EA Sports (Electronic Arts). The series, which debuted in 1993, is
known for its realistic and authentic football experience, with a focus on gameplay, strategy, and simulation.

The game is based on the National Football League (NFL) and features current and retired players, as well as fictional characters created by 
the developers. The series has undergone significant updates and improvements over the years, incorporating new technologies and features to 
enhance gameplay and realism.

Here are some key features and characteristics of NFL Madden:

1. **Gameplay simulation**: The game is designed to simulate the gameplay and strategies of the NFL, including player movements, tackling,   
throwing, and running.
2. **Authentic player roster**: The game features a large roster of NFL players, including current and retired players, as well as players   
from college and high school football.
3. **Customization options**: Players can customize their players with various attributes, such as height, weight, speed, strength, and more.
4. **Season modes**: The game features a variety of season modes, including regular seasons, playoffs, and preseason games.
5. **Tactics and play calling**: Players can use various tactics and play calling options to gain an advantage over opponents.
6. **Reputation system**: Players can build their reputation by winning games, which affects their standing with coaches and teammates.      
7. **Realistic graphics and sound**: The game features detailed graphics and sound effects'''
)

SAMPLE_PARAGRAPH1 = '''
Drake Maye is an American football quarterback who played college football at North Carolina. Here are the economic nuances and impact of his
career during his time at the University of North Carolina:

**Economic Nuances:**

1. **College Football's Economic Significance:** College football is a multi-billion-dollar industry, generating revenue from television     
deals, sponsorships, ticket sales, and merchandise. The University of North Carolina's football program, in particular, has a significant    
economic impact on the state of North Carolina.
2. **Player Contracts:** As a top college football player, Maye's salary and endorsements can generate significant revenue for his family and
the university. His contracts can increase in value as he gains experience and becomes a more prominent player.
3. **Endorsement Deals:** Maye's endorsement deals can provide a significant source of revenue for his family and the university. As a       
high-profile athlete, Maye can attract endorsement deals from major brands, such as Nike, Under Armour, and Gatorade, generating substantial 
income.

**Impact on the University:**

1. **Boost to Tourism:** Drake Maye's presence at the University of North Carolina can attract fans and visitors to the state, generating    
revenue for local hotels, restaurants, and other businesses.
2. **Increased Athletics Revenue:** Maye's success on the field can lead to increased revenue from ticket sales, merchandise, and other      
athletic department revenue streams.
3. **University Brand:** Maye's popularity can enhance the university'''

MODULUS = 1024


def main() -> int:
    console = Console(highlight=False)
    scorer = get_scorer()

    console.print(
        Panel.fit(
            f"[bold]classifier[/] {scorer.model_id}\n"
            f"[bold]cutoff[/] {SCORE_CUTOFF:g}  ·  [bold]|V|[/] {len(VOCABULARY)}",
            title="Label classification",
        )
    )
    console.rule("Sample text", style="dim")
    console.print(SAMPLE_PARAGRAPH)

    scores = classify_text(SAMPLE_PARAGRAPH)
    active_labels = active_labels_from_attributes(
        derive_attributes(SAMPLE_PARAGRAPH, MODULUS, log_scores=False),
        MODULUS,
    )
    active_set = set(active_labels)

    table = Table(title=f"Label scores (multi-label active if score >= {SCORE_CUTOFF:g})")
    table.add_column("label", style="dim")
    table.add_column("score", justify="right")
    table.add_column(f"active(>={SCORE_CUTOFF:g})", justify="center")

    for label in VOCABULARY:
        yn = "yes" if label in active_set else "no"
        style = "bold green" if yn == "yes" else None
        table.add_row(label, f"{scores[label]:.4f}", yn, style=style)

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Active labels:[/] {active_labels if active_labels else '(none)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
