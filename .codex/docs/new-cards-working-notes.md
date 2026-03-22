# New Cards Working Notes

Status: OCR-derived working manifest for cards still in `~/Desktop/tcg-card-imgs/unfinished`.

Purpose: keep extracted card info in-repo while implementing cards in batches. Treat OCR values as hints until verified against the image.

Conventions:
- `card_code_guess` removes the `S1-` prefix. If the file uses an `A` code, that likely indicates an alternate-art / special-rarity DB variant and may still map to the base engine card code without the trailing `A`.
- `ocr_nums` is raw OCR for the numeric badges and is not yet normalized into cost / gate / attack / health.
- Effect text and subtype lines are OCR-derived and may contain small transcription errors.

## AZK01

### S1-AZK01-021
- file: `S1-AZK01-021_Mizuto_E_C_die.jpg`
- card_code_guess: `AZK01-021`
- name_guess: `Mizuto`
- type_guess: `ENTITY`
- subtype_guess: `Wavecaller`
- ocr_nums: `1`
- effect_guess: On Play Look at the top 5 cards of your deck, reveal up to 1 Driftward subtype card and add it to your hand, then bottom deck the rest in any order.

### S1-AZK01-022
- file: `S1-AZK01-022_Mirage-Frog_E_UC_die.jpg`
- card_code_guess: `AZK01-022`
- name_guess: `Mirage Frog`
- type_guess: `ENTITY`
- subtype_guess: `Frog, Rippleborn, Watercrafting`
- ocr_nums: `2, 11, 2`
- effect_guess: On Play You may discard 1: Return an entity with a cost of 2 or less in any Garden to its owner's hand.

### S1-AZK01-023
- file: `S1-AZK01-023_Maho_E_C_die.jpg`
- card_code_guess: `AZK01-023`
- name_guess: `Maho`
- type_guess: `ENTITY`
- subtype_guess: `Wavecaller`
- ocr_nums: `3, 1, 2`
- effect_guess: In Garden Only Ability End of Your Turn If you have 2 or less cards in your hand, draw 1. 1

### S1-AZK01-024
- file: `S1-AZK01-024_Fumiko_E_UC_die.jpg`
- card_code_guess: `AZK01-024`
- name_guess: `Fumiko`
- type_guess: `ENTITY`
- subtype_guess: `Painter, Watercrafting`
- ocr_nums: `3, 11, 2`
- effect_guess: On Play You may return an entity in your Garden to your hand: Play an entity with a cost of 2 or less from your hand.

### S1-AZK01-025
- file: `S1-AZK01-025_Lighthouse-Keeper_E_UC_die.jpg`
- card_code_guess: `AZK01-025`
- name_guess: `Lighthouse Keeper`
- type_guess: `ENTITY`
- subtype_guess: `Driftward, Elder`
- ocr_nums: `3`
- effect_guess: Defender (If this card is in the Garden, you may tap it to redirect an attack to this card) This card cannot take damage from card effects.

### S1-AZK01-026
- file: `S1-AZK01-026_Moonlit-Crane_E_C_die.jpg`
- card_code_guess: `AZK01-026`
- name_guess: `Moonlit Crane`
- type_guess: `ENTITY`
- subtype_guess: `Bird, Driftward, Rippleborn`
- ocr_nums: `5, 2, 2, 2`
- effect_guess: KResponse Once/Turn You may discard 1: Reduce a leader's or an entity's attack by 1 until the end of the turn. 2

### S1-AZK01-027
- file: `S1-AZK01-027_Kaiya-Mizumi_E_R_die.jpg`
- card_code_guess: `AZK01-027`
- name_guess: `Kaiya Mizumi`
- type_guess: `ENTITY`
- subtype_guess: `Wavecaller`
- ocr_nums: `2, 3`
- effect_guess: On Play You may discard up to 4 cards: Draw cards equal to the number of cards you discarded.

### S1-AZK01-028
- file: `S1-AZK01-028_Soryu-no-Rin_E_SR_die.jpg`
- card_code_guess: `AZK01-028`
- name_guess: `Sory no Rin`
- type_guess: `NIT`
- subtype_guess: `Watercrafting, Wavecaller`
- ocr_nums: `8, 7`
- effect_guess: On Play You must discard your hand: Return all other entities in each player's Garden to their owner's hands.

### S1-AZK01-029
- file: `S1-AZK01-029_Aquatic-Veil_S_R_die.jpg`
- card_code_guess: `AZK01-029`
- name_guess: `Aquatic Veil`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: *Response Discard 2: Reduce a leader's or an entity's- attack by 3 until the end of the turn. "Let the currents sap your will and weaken your strike."

### S1-AZK01-030
- file: `S1-AZK01-030_Lotus-of-Paradise_S_C_die.jpg`
- card_code_guess: `AZK01-030`
- name_guess: `Lotus of Paradise`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: *Response Untap up to 2 IKZ. *The Lotus of Paradise blooms where peace reigns its petals unlocking the soul's true light."

### S1-AZK01-031
- file: `S1-AZK01-031_Tidal-Insight_S_UC_die.jpg`
- card_code_guess: `AZK01-031`
- name_guess: `Tidal Insight`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: Main • Look at the top 3 cards of your deck, reveal up to 1 @ card and add it to your hand, then put the rest on the top or bottom of your deck in any order.

### S1-AZK01-032
- file: `S1-AZK01-032_Rippling-Recall_S_C_die.jpg`
- card_code_guess: `AZK01-032`
- name_guess: `Rippling Recall`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `3`
- effect_guess: Main Return an entity with a cost of ® or more in your Garden to your hand: Return up to 1 entity with a cost of 4 or less in your opponent's Garden to its owner's hand.

## Verified Batch: AZK01-021 to AZK01-032

These entries were manually verified from the card images and supersede the OCR guesses above.

- `AZK01-021` Mizuto
  - `ENTITY`, `WATER`, rarity `C`
  - cost `1`, gate `1`, atk `1`, hp `1`
  - subtypes: `Wavecaller`
  - effect: `[On Play] Look at the top 5 cards of your deck, reveal up to 1 Driftward subtype card and add it to your hand, then bottom deck the rest in any order.`
- `AZK01-022` Mirage Frog
  - `ENTITY`, `WATER`, rarity `UC`
  - cost `2`, gate `1`, atk `1`, hp `2`
  - subtypes: `Frog`, `Rippleborn`, `Watercrafting`
  - effect: `[On Play] You may discard 1: Return an entity with a cost of 2 or less in any Garden to its owner's hand.`
- `AZK01-023` Maho
  - `ENTITY`, `WATER`, rarity `C`
  - cost `3`, gate `1`, atk `1`, hp `2`
  - subtypes: `Wavecaller`
  - effect: `[In Garden Only Ability][End of Your Turn] If you have 2 or less cards in your hand, draw 1.`
- `AZK01-024` Fumiko
  - `ENTITY`, `WATER`, rarity `UC`
  - cost `3`, gate `1`, atk `1`, hp `2`
  - subtypes: `Painter`, `Watercrafting`
  - effect: `[On Play] You may return an entity in your Garden to your hand: Play an entity with a cost of 2 or less from your hand.`
- `AZK01-025` Lighthouse Keeper
  - `ENTITY`, `WATER`, rarity `UC`
  - cost `4`, gate `0`, atk `0`, hp `3`
  - subtypes: `Driftward`, `Elder`
  - keywords/effect: `Defender`; `This card cannot take damage from card effects.`
- `AZK01-026` Moonlit Crane
  - `ENTITY`, `WATER`, rarity `C`
  - cost `5`, gate `2`, atk `2`, hp `2`
  - subtypes: `Bird`, `Driftward`, `Rippleborn`
  - effect: `[Response][Once/Turn] You may discard 1: Reduce a leader's or an entity's attack by 1 until the end of the turn.`
- `AZK01-027` Kaiya Mizumi
  - `ENTITY`, `WATER`, rarity `R`
  - cost `6`, gate `2`, atk `3`, hp `3`
  - subtypes: `Wavecaller`
  - effect: `[On Play] You may discard up to 4 cards: Draw cards equal to the number of cards you discarded.`
- `AZK01-028` Soryu no Rin
  - `ENTITY`, `WATER`, rarity `SR`
  - cost `8`, gate `2`, atk `7`, hp `7`
  - subtypes: `Watercrafting`, `Wavecaller`
  - effect: `[On Play] You must discard your hand: Return all other entities in each player's Garden to their owner's hands.`
- `AZK01-029` Aquatic Veil
  - `SPELL`, `WATER`, rarity `R`
  - cost `0`
  - subtype: `Aquashield`
  - effect: `[Response] Discard 2: Reduce a leader's or an entity's attack by 3 until the end of the turn.`
- `AZK01-030` Lotus of Paradise
  - `SPELL`, `WATER`, rarity `C`
  - cost `1`
  - subtypes: `Lotus`, `Watercrafting`
  - effect: `[Response] Untap up to 2 IKZ.`
- `AZK01-031` Tidal Insight
  - `SPELL`, `WATER`, rarity `UC`
  - cost `1`
  - subtype: `Sirencall`
  - effect: `[Main] Look at the top 3 cards of your deck, reveal up to 1 Water card and add it to your hand, then put the rest on the top or bottom of your deck in any order.`
- `AZK01-032` Rippling Recall
  - `SPELL`, `WATER`, rarity `C`
  - cost `3`
  - subtype: `Watercrafting`
  - effect: `[Main] Return an entity with a cost of 2 or more in your Garden to your hand: Return up to 1 entity with a cost of 4 or less in your opponent's Garden to its owner's hand.`

### S1-AZK01-033
- file: `S1-AZK01-033_Elder-Hoshin_E_C_die.jpg`
- card_code_guess: `AZK01-033`
- name_guess: `Elder Hoshin`
- type_guess: `ENTITY`
- subtype_guess: `Elder, Monk, Stormcaller`
- ocr_nums: `1`
- effect_guess: On Play Look at the top 5 cards of your deck, reveal up to 1 Steelborn subtype card and add it to your hand, then bottom deck the rest in any order.

### S1-AZK01-034
- file: `S1-AZK01-034_Kira_E_C_die.jpg`
- card_code_guess: `AZK01-034`
- name_guess: `Kira`
- type_guess: `ENTITY`
- subtype_guess: `Voltguard`
- ocr_nums: `2, 1`
- effect_guess: In Alley Only Ability Whenever an entity in your Garden is attacked, you may swap that entity and this card, then make this card the new attack target.

### S1-AZK01-035
- file: `S1-AZK01-035_Raimaru-the-Stolen_E_C_die.jpg`
- card_code_guess: `AZK01-035`
- name_guess: `Raimaru the Stolen`
- type_guess: `ENTITY`
- subtype_guess: `Cat, Statue, Voltguard`
- ocr_nums: `3, 2025`
- effect_guess: Defender (If this card is in the Garden, you may tap it to redirect an attack to this card) This card can be played as a Response

### S1-AZK01-036
- file: `S1-AZK01-036_Denmu_E_UC_die.jpg`
- card_code_guess: `AZK01-036`
- name_guess: `Denmu`
- type_guess: `ENTITY`
- subtype_guess: `Bat, Shockcoil`
- ocr_nums: `3, 2, 2`
- effect_guess: When Attacked The attacking leader or entity becomes Shocked (Does not untap during its next untap phase).

### S1-AZK01-037
- file: `S1-AZK01-037_Stormcaller-Tenkichi_E_R_die.jpg`
- card_code_guess: `AZK01-037`
- name_guess: `Stormcaller Tenkichi`
- type_guess: `ENTITY`
- subtype_guess: `Riftwalk, Stormcaller`
- ocr_nums: `4, 2`
- effect_guess: Charge (Can attack the same turn it enters the Garden) This card can attack untapped or tapped entities in your opponent's Alley.

### S1-AZK01-038
- file: `S1-AZK01-038_Riven-Flashborne_E_R_die.jpg`
- card_code_guess: `AZK01-038`
- name_guess: `Riven Flashborne`
- type_guess: `ENTITY`
- subtype_guess: `Riftwalk`
- ocr_nums: `3, 2`
- effect_guess: This card can attack untapped or tapped entities in your opponent's Alley.

### S1-AZK01-039
- file: `S1-AZK01-039_Piko_E_C_die.jpg`
- card_code_guess: `AZK01-039`
- name_guess: `Piko`
- type_guess: `ENTITY`
- subtype_guess: `Cat, Steelborn`
- ocr_nums: `3`
- effect_guess: When Equipped Gain Charge (Can attack the same turn it enters the Garden).

### S1-AZK01-040
- file: `S1-AZK01-040_Black-Jade-Vault-Master_E_UC_die.jpg`
- card_code_guess: `AZK01-040`
- name_guess: `Black Jade Vault Master`
- type_guess: `UNKNOWN`
- subtype_guess: `Black Jade, Tiger, Voltguard`
- ocr_nums: `5, 2, 3`
- effect_guess: Defender (f this card is in the Garden, you may tap it to redirect an attack to this card) When Attacked Deal up to 1 damage to a leader.

### S1-AZK01-041
- file: `S1-AZK01-041_Wu-Cha_E_SR_die.jpg`
- card_code_guess: `AZK01-041`
- name_guess: `Wu Cha`
- type_guess: `UNKNOWN`
- subtype_guess: `Elder, Monk`
- ocr_nums: `6, 4`
- effect_guess: Response Until the end of the turn, the next time your opponent would trigger an On Play ability, instead choose up to 1 damage to a leader or entity in any Garden.

### S1-AZK01-042
- file: `S1-AZK01-042_Thunderclap_S_SR_die.jpg`
- card_code_guess: `AZK01-042`
- name_guess: `Thunderclap`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `5`
- effect_guess: Main Deal 2 damage to an entity or a leader in any Garden. That leader or entity becomes Shocked. 2

### S1-AZK01-043
- file: `S1-AZK01-043_Stormglass-Daggers_W_C_die.jpg`
- card_code_guess: `AZK01-043`
- name_guess: `Stormglass Daggers`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: On Play Draw 1, then discard 1.

### S1-AZK01-044
- file: `S1-AZK01-044_Lightning-Kanabo_W_R_die.jpg`
- card_code_guess: `AZK01-044`
- name_guess: `Lightning Kanabo`
- type_guess: `WEAPON`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: When Attacking If the attacking card is a Leader, you may untap 1 IKZ.

## Verified Batch: AZK01-033 to AZK01-044

- `AZK01-033` Elder Hoshin
  - `ENTITY`, `LIGHTNING`, rarity `C`
  - cost `1`, gate `1`, attack `1`, health `1`
  - subtypes: `Elder`, `Monk`, `Stormcaller`
  - effect: `[On Play] Look at the top 5 cards of your deck, reveal up to 1 Steelborn subtype card and add it to your hand, then bottom deck the rest in any order.`
- `AZK01-034` Kira
  - `ENTITY`, `LIGHTNING`, rarity `C`
  - cost `2`, gate `1`, attack `1`, health `1`
  - subtype: `Voltguard`
  - effect: `[In Alley Only Ability] Whenever an entity in your Garden is attacked, you may swap that entity and this card, then make this card the new attack target.`
- `AZK01-035` Raimaru the Stolen
  - `ENTITY`, `LIGHTNING`, rarity `C`
  - cost `3`, gate `0`, attack `0`, health `1`
  - keywords: `Defender`
  - subtypes: `Cat`, `Statue`, `Voltguard`
  - effect: `Defender. This card can be played as a [Response].`
- `AZK01-036` Denmu
  - `ENTITY`, `LIGHTNING`, rarity `UC`
  - cost `3`, gate `1`, attack `2`, health `2`
  - subtypes: `Bat`, `Shockcoil`
  - effect: `[When Attacked] The attacking leader or entity becomes Shocked. (Does not untap during its next untap phase.)`
- `AZK01-037` Stormcaller Tenkichi
  - `ENTITY`, `LIGHTNING`, rarity `R`
  - cost `4`, gate `0`, attack `3`, health `2`
  - keywords: `Charge`
  - subtypes: `Riftwalk`, `Stormcaller`
  - effect: `Charge. This card can attack untapped or tapped entities in your opponent's Alley.`
- `AZK01-038` Riven Flashborne
  - `ENTITY`, `LIGHTNING`, rarity `R`
  - cost `3`, gate `1`, attack `2`, health `2`
  - subtype: `Riftwalk`
  - effect: `This card can attack untapped or tapped entities in your opponent's Alley.`
- `AZK01-039` Piko
  - `ENTITY`, `LIGHTNING`, rarity `C`
  - cost `4`, gate `0`, attack `2`, health `3`
  - subtypes: `Cat`, `Steelborn`
  - effect: `[When Equipped] Gain Charge.`
- `AZK01-040` Black Jade Vault Master
  - `ENTITY`, `LIGHTNING`, rarity `UC`
  - cost `5`, gate `2`, attack `1`, health `3`
  - keywords: `Defender`
  - subtypes: `Black Jade`, `Tiger`, `Voltguard`
  - effect: `Defender. [When Attacked] Deal up to 1 damage to a leader.`
- `AZK01-041` Wu Cha
  - `ENTITY`, `LIGHTNING`, rarity `SR`
  - cost `8`, gate `2`, attack `7`, health `7`
  - subtypes: `Fox Fire`, `Steelborn`
  - effect: `[In Garden Only Ability][Once/Turn][Main] 1 IKZ: Equip up to 2 Weapon cards with a cost of 2 or less from your discard pile to this card.`
- `AZK01-042` Thunderclap
  - `SPELL`, `LIGHTNING`, rarity `SR`
  - cost `5`
  - subtype: `Stormcaller`
  - effect: `[Main] Deal 3 damage, 2 damage, and 1 damage to 3 different entities in your opponent's Garden.`
- `AZK01-043` Stormglass Daggers
  - `WEAPON`, `LIGHTNING`, rarity `C`
  - cost `2`, attack `1`
  - subtypes: `Dagger`, `Riftwalk`
  - effect: `When equipped to a leader, the leader can attack untapped or tapped entities in your opponent's Alley.`
- `AZK01-044` Lightning Kanabo
  - `WEAPON`, `LIGHTNING`, rarity `R`
  - cost `3`, attack `2`
  - subtype: `Shockcoil`
  - effect: `[Once/Turn] Whenever a leader or an entity equipped with this card deals combat damage to an opponent's card, that card becomes Shocked. (Does not untap during its next untap phase.)`

### S1-AZK01-045
- file: `S1-AZK01-045_Treetop-Scout_E_C_die.jpg`
- card_code_guess: `AZK01-045`
- name_guess: `Treetop Scout`
- type_guess: `ENTITY`
- subtype_guess: `Scavenger`
- ocr_nums: `1, 1`
- effect_guess: In Alley Only Ability Whenever one or more cards are returned to a player's hand from the Garden, you may heal 1 to your leader.

### S1-AZK01-046
- file: `S1-AZK01-046_Mina-the-Geomancer_E_UC_die.jpg`
- card_code_guess: `AZK01-046`
- name_guess: `Mina the Geomancer`
- type_guess: `ENTITY`
- subtype_guess: `Elder, Stone Shaper`
- ocr_nums: `2, 1`
- effect_guess: Once/Turn In Alley Only Ability If one or more cards would be returned to your hand from your Garden, you may discard 1: instead, return those cards to your hand and put this card into your Garden.

### S1-AZK01-047
- file: `S1-AZK01-047_Shiko-the-Priestess_E_UC_die.jpg`
- card_code_guess: `AZK01-047`
- name_guess: `Shiko the Priestess`
- type_guess: `UNKNOWN`
- subtype_guess: `Elder, Shrineguard`
- ocr_nums: `3, 2`
- effect_guess: On Play Heal 1 to your leader.

### S1-AZK01-048
- file: `S1-AZK01-048_Kale_E_C_die.jpg`
- card_code_guess: `AZK01-048`
- name_guess: `Kale`
- type_guess: `ENTITY`
- subtype_guess: `Brawler, Dawnling`
- ocr_nums: `3, 2`
- effect_guess: When Attacking You may return 1 entity with a cost of 2 or less in your Garden to your hand: Return this card to your hand.

### S1-AZK01-049
- file: `S1-AZK01-049_Lone-Journeyman_E_C_die.jpg`
- card_code_guess: `AZK01-049`
- name_guess: `Lone Journeyman`
- type_guess: `ENTITY`
- subtype_guess: `Stone Shaper`
- ocr_nums: `4, 2`
- effect_guess: This card has +1 attack for each different subtype among entities in your Garden.

### S1-AZK01-050
- file: `S1-AZK01-050_Shroom-Tender_E_R_die.jpg`
- card_code_guess: `AZK01-050`
- name_guess: `Shroom Tender`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Elder`
- ocr_nums: `4, 2, 3`
- effect_guess: On Play Look at the top 4 cards of your deck, reveal up to 1 [Earth] card and add it to your hand, then bottom deck the rest in any order.

### S1-AZK01-051
- file: `S1-AZK01-051_Chillax_E_UC_die.jpg`
- card_code_guess: `AZK01-051`
- name_guess: `Chillax`
- type_guess: `ENTITY`
- subtype_guess: `Frog, Scavenger, Yonkai`
- ocr_nums: `4, 3`
- effect_guess: When Attacked The attacking leader or entity gets -2 attack until the end of the tum.

### S1-AZK01-052
- file: `S1-AZK01-052_Yojin_E_UC_die.jpg`
- card_code_guess: `AZK01-052`
- name_guess: `Yojin`
- type_guess: `ENTITY`
- subtype_guess: `Cloudstrider, Scavenger`
- ocr_nums: `3, 2`
- effect_guess: In Alley Only Ability Whenever an entity in your opponent's Garden is returned to their hand or destroyed, you may play this card.

### S1-AZK01-053
- file: `S1-AZK01-053_Geodust-Smuggler_E_R_die.jpg`
- card_code_guess: `AZK01-053`
- name_guess: `Geodust Smuggler`
- type_guess: `ENTITY`
- subtype_guess: `Bandit, Scavenger`
- ocr_nums: `5, 3`
- effect_guess: Response Discard 1: Return up to 1 entity in your Garden to your hand. 1

### S1-AZK01-054
- file: `S1-AZK01-054_Teb-Fea_E_SR_die.jpg`
- card_code_guess: `AZK01-054`
- name_guess: `Teb Fea`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Elder`
- ocr_nums: `6, 4, 4`
- effect_guess: If an entity would be returned to your hand from your Garden, play that entity into your Garden instead. 4

### S1-AZK01-055
- file: `S1-AZK01-055_Earth-Orb_S_C_die.jpg`
- card_code_guess: `AZK01-055`
- name_guess: `Earth Orb`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: [Response] Discard 1: Give a leader or entity in any Garden +2 attack until the end of the turn.

## Verified Batch: AZK01-045 to AZK01-055

- `AZK01-045` Treetop Scout
  - `ENTITY`, `EARTH`, rarity `C`
  - cost `1`, gate `1`, attack `1`, health `1`
  - subtype: `Scout`
  - effect: `[On Play] Look at the top 5 cards of your deck, reveal up to 1 Obsidian subtype card and add it to your hand, then bottom deck the rest in any order.`
- `AZK01-046` Mina the Geomancer
  - `ENTITY`, `EARTH`, rarity `UC`
  - cost `1`, gate `0`, attack `1`, health `1`
  - subtype: `Earthfury`
  - effect: `When this card enters the Garden, tap it. This entity cannot be untapped. [In Garden Only Ability] [Start of Your Turn] Deal up to 1 damage to a leader.`
- `AZK01-047` Shiko the Priestess
  - `ENTITY`, `EARTH`, rarity `UC`
  - cost `2`, gate `1`, attack `1`, health `2`
  - subtype: `Earthwarden`
  - effect: `[Once/Turn] [When Attacking] Heal 1 to your leader.`
- `AZK01-048` Kale
  - `ENTITY`, `EARTH`, rarity `C`
  - cost `3`, gate `2`, attack `1`, health `2`
  - subtype: `Obsidian`
  - effect: `Carapace 1. (Reduce damage from all sources by 1. Carapace stacks.)`
- `AZK01-049` Lone Journeyman
  - `ENTITY`, `EARTH`, rarity `C`
  - cost `3`, gate `2`, attack `1`, health `2`
  - keyword: `Defender`
  - subtype: `Earthwarden`
  - effect: `Defender.`
- `AZK01-050` Shroom Tender
  - `ENTITY`, `EARTH`, rarity `R`
  - cost `4`, gate `2`, attack `2`, health `2`
  - subtypes: `Beanz`, `Earthwarden`
  - effect: `[On Play] Heal 2 to your leader.`
- `AZK01-051` Chillax
  - `ENTITY`, `EARTH`, rarity `UC`
  - cost `4`, gate `2`, attack `1`, health `3`
  - keyword: `Defender`
  - subtypes: `Earthwarden`, `Sloth`
  - effect: `[When Attacked] You may give another entity in your Garden +1 health until the end of the turn.`
- `AZK01-052` Yojin
  - `ENTITY`, `EARTH`, rarity `UC`
  - cost `5`, gate `2`, attack `2`, health `4`
  - subtype: `Earthwarden`
  - effect: `If the number of entities in your Garden is less than the number of entities in your opponent's Garden, this card has Defender.`
- `AZK01-053` Geodust Smuggler
  - `ENTITY`, `EARTH`, rarity `R`
  - cost `7`, gate `3`, attack `4`, health `5`
  - subtype: `Obsidian`
  - effect: `[In Garden Only Ability] As long as this card is in play, all other entities in your Garden have +1 health.`
- `AZK01-054` Teb Fea
  - `ENTITY`, `EARTH`, rarity `SR`
  - cost `8`, gate `2`, attack `7`, health `7`
  - keyword: `Godmode`
  - subtype: `Obsidian`
  - effect: `Godmode. (Cannot leave the field by taking damage or by card effects.)`
- `AZK01-055` Earth Orb
  - `SPELL`, `EARTH`, rarity `C`
  - cost `2`
  - subtypes: `Earthfury`, `Orb`
  - effect: `[Response] Deal up to 1 damage to a leader or an entity in your opponent's Garden and reduce that card's attack by 1 until the end of the turn.`

### S1-AZK01-056
- file: `S1-AZK01-056_Glass-Blower-Hokuto_E_C_die.jpg`
- card_code_guess: `AZK01-056`
- name_guess: `Glass Blower Hokuto`
- type_guess: `ENTITY`
- subtype_guess: `Blazeartisan`
- ocr_nums: `1, 1`
- effect_guess: On Play Look at the top 5 cards of your deck, reveal up to 1 Pyreskin subtype card and add it to your hand, then bottom deck the rest in any order.

### S1-AZK01-057
- file: `S1-AZK01-057_Lounge-Siren-Saeko_E_C_die.jpg`
- card_code_guess: `AZK01-057`
- name_guess: `Lounge Siren Saeko`
- type_guess: `ENTITY`
- subtype_guess: `Siren`
- ocr_nums: `2, 1`
- effect_guess: When Attacking You may deal 1 damage to your leader: Draw 1.

### S1-AZK01-058
- file: `S1-AZK01-058_Black-Jade-Warlord_E_C_die.jpg`
- card_code_guess: `AZK01-058`
- name_guess: `Black Jade Warlord`
- type_guess: `ENTITY`
- subtype_guess: `Black Jade, Commander, Pyreskin`
- ocr_nums: `2, 1, 2`
- effect_guess: When Attacking Until the end of the tum, an entity in your Garden gets +1 attack.

### S1-AZK01-059
- file: `S1-AZK01-059_Spice_E_UC_die.jpg`
- card_code_guess: `AZK01-059`
- name_guess: `Spice`
- type_guess: `ENTITY`
- subtype_guess: `Blazeartisan, Pyreskin`
- ocr_nums: `3, 2`
- effect_guess: On Play You must deal 1 damage to your leader: Draw 1.

### S1-AZK01-060
- file: `S1-AZK01-060_Scarlett_E_R_die.jpg`
- card_code_guess: `AZK01-060`
- name_guess: `Scarlett`
- type_guess: `ENTITY`
- subtype_guess: `Siren`
- ocr_nums: `2, 3`
- effect_guess: Once/Turn In Garden Only Ability Deal 1 damage to your leader: This card gets +1 attack until the end of the turn.

### S1-AZK01-061
- file: `S1-AZK01-061_Firebrand-Renji_E_UC_die.jpg`
- card_code_guess: `AZK01-061`
- name_guess: `Firebrand Renji`
- type_guess: `UNKNOWN`
- subtype_guess: `Blazeartisan, Dawnling`
- ocr_nums: `4, 2`
- effect_guess: Once/Turn When Equipped You may deal 1 damage to your leader: Draw 1.

### S1-AZK01-062
- file: `S1-AZK01-062_Pekiro_E_R_die.jpg`
- card_code_guess: `AZK01-062`
- name_guess: `Pekiro`
- type_guess: `ENTITY`
- subtype_guess: `Bird, Pyreskin`
- ocr_nums: `4, 2`
- effect_guess: When Attacking If you have 2 or less cards in your hand, this card gets +1 attack until the end of the turn.

### S1-AZK01-063
- file: `S1-AZK01-063_Enzo_E_UC_die.jpg`
- card_code_guess: `AZK01-063`
- name_guess: `Enzo`
- type_guess: `ENTITY`
- subtype_guess: `Companion, Elder, Pyreskin`
- ocr_nums: `5, 3`
- effect_guess: On Play You must deal 1 damage to your leader: You may play an entity with a cost of 2 or less from your hand. 3

### S1-AZK01-064
- file: `S1-AZK01-064_Zero_E_SR_die.jpg`
- card_code_guess: `AZK01-064`
- name_guess: `Zero`
- type_guess: `ENTITY`
- subtype_guess: `Blazeartisan, Pyreskin`
- ocr_nums: `7, 4, 4`
- effect_guess: In Garden Only Ability If your leader would take damage, this card takes that damage instead. Once/Turn When This Card Takes Damage Return up to 1 entity in your Garden to your hand. 4

### S1-AZK01-065
- file: `S1-AZK01-065_Fire-Orb_S_C_die.jpg`
- card_code_guess: `AZK01-065`
- name_guess: `Fire Orb`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Main Deal 1 damage to your leader: Deal 1 damage to a leader or entity in any Garden.

### S1-AZK01-066
- file: `S1-AZK01-066_Firestorm_S_UC_die.jpg`
- card_code_guess: `AZK01-066`
- name_guess: `Firestorm`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: Main Deal 1 damage to each leader and entity in any Garden.

## Verified Batch: AZK01-056 to AZK01-066

- `AZK01-056` Glass Blower, Hokuto
  - `ENTITY`, `FIRE`, rarity `C`
  - cost `1`, gate `1`, attack `1`, health `1`
  - subtypes: `Artisan`, `Firemancer`
  - effect: `[On Play] Look at the top 5 cards of your deck, reveal up to 1 Scorchweaver subtype card and add it to your hand, then bottom deck the rest in any order.`
- `AZK01-057` Lounge Siren, Saeko
  - `ENTITY`, `FIRE`, rarity `C`
  - cost `1`, gate `1`, attack `1`, health `1`
  - subtype: `Scorchweaver`
  - effect: `[In Garden Only Ability] [Start of Your Turn] You must deal 1 damage to an entity in your Garden and 1 damage to an entity in your opponent's Garden.`
- `AZK01-058` Black Jade Warlord
  - `ENTITY`, `FIRE`, rarity `C`
  - cost `2`, gate `1`, attack `2`, health `1`
  - subtypes: `Black Jade`, `Blazerker`, `Invigorate`
  - effect: `[After Attacking] You may sacrifice this card: Give a leader or an entity in your Garden +2 attack until the end of the turn.`
- `AZK01-059` Spice
  - `ENTITY`, `FIRE`, rarity `UC`
  - cost `2`, gate `1`, attack `1`, health `2`
  - subtypes: `Chef`, `Pyreskin`, `Red Panda`
  - effect: `[Once/Turn] Whenever this card takes damage, give another entity in your Garden +1 attack until the end of the turn.`
- `AZK01-060` Scarlett
  - `ENTITY`, `FIRE`, rarity `R`
  - cost `5`, gate `2`, attack `3`, health `2`
  - keyword: `Charge`
  - subtype: `Blazerker`
  - effect: `[When Attacking] You may give this card Infiltrate and +1 attack until the end of the turn. If you do, you must sacrifice this card at the end of the turn.`
- `AZK01-061` Firebrand Renji
  - `ENTITY`, `FIRE`, rarity `UC`
  - cost `3`, gate `1`, attack `1`, health `3`
  - subtype: `Pyreskin`
  - effect: `[Once/Turn] Whenever this card takes damage from 3 different damage sources in one turn, deal up to 3 damage to a leader or an entity in any Garden.`
- `AZK01-062` Pekiro
  - `ENTITY`, `FIRE`, rarity `R`
  - cost `3`, gate `2`, attack `2`, health `1`
  - subtypes: `Red Panda`, `Pyreskin`
  - effect: `[Once/Turn] Whenever this card would take damage from an ability or spell, you may redirect that damage to another entity in any Garden.`
- `AZK01-063` Enzo
  - `ENTITY`, `FIRE`, rarity `UC`
  - cost `6`, gate `2`, attack `3`, health `3`
  - subtypes: `Chef`, `Scorchweaver`
  - effect: `[On Play] If this card is played in the Garden, deal up to 3 damage to a leader or an entity in any Garden.`
- `AZK01-064` Zero
  - `ENTITY`, `FIRE`, rarity `SR`
  - cost `8`, gate `2`, attack `7`, health `7`
  - subtypes: `Black Jade`, `Scorchweaver`
  - effect: `When this card enters the Garden, deal 2 damage to all entities in each player's Garden.`
- `AZK01-065` Fire Orb
  - `SPELL`, `FIRE`, rarity `C`
  - cost `4`
  - subtypes: `Orb`, `Scorchweaver`
  - effect: `[Main] Deal 3 damage to your leader: Deal up to 5 damage to a leader or an entity in any Garden.`
- `AZK01-066` Firestorm
  - `SPELL`, `FIRE`, rarity `UC`
  - cost `5`
  - subtype: `Scorchweaver`
  - effect: `[Main] Deal 2 damage to all leaders and entities in each player's Garden.`

### S1-AZK01-067
- file: `S1-AZK01-067_Frida_E_C_die.jpg`
- card_code_guess: `AZK01-067`
- name_guess: `Frida`
- type_guess: `ENTITY`
- subtype_guess: `Companion`
- ocr_nums: `1, 2`
- effect_guess: On Play Look at the top 5 cards of your deck, reveal up to 1 Beanz subtype card and add it to your hand, then bottom deck the rest in any order.

### S1-AZK01-068
- file: `S1-AZK01-068_Pip_E_C_die.jpg`
- card_code_guess: `AZK01-068`
- name_guess: `Pip`
- type_guess: `ENTITY`
- subtype_guess: `Beanz, Companion`
- ocr_nums: `2, 1`
- effect_guess: On Play Choose another entity in your Garden. Switch its attack and health until the end of the turn.

### S1-AZK01-069
- file: `S1-AZK01-069_Link_E_C_die.jpg`
- card_code_guess: `AZK01-069`
- name_guess: `Link`
- type_guess: `ENTITY`
- subtype_guess: `Beanz, Companion`
- ocr_nums: `2, 1`
- effect_guess: When Attacking Reveal the top card of your deck. If it's an entity, this card gets +1 attack until the end of the turn.

### S1-AZK01-070
- file: `S1-AZK01-070_Mocking-Dummy_E_C_die.jpg`
- card_code_guess: `AZK01-070`
- name_guess: `Mocking Dummy`
- type_guess: `ENTITY`
- subtype_guess: `Dummy`
- ocr_nums: `2, 2`
- effect_guess: Defender (If this card is in the Garden, you may tap it to redirect an attack to this card) Deal 1 damage to your leader on your next turn. 2

### S1-AZK01-071
- file: `S1-AZK01-071_Alley-Fetchduck_E_UC_die.jpg`
- card_code_guess: `AZK01-071`
- name_guess: `Alley Fetchduck`
- type_guess: `ENTITY`
- subtype_guess: `Bird, Companion`
- ocr_nums: `2, 1, 2`
- effect_guess: When This Card is Returned to Hand or Destroyed Look at the top 5 cards of your deck, reveal up to 1 Companion or Elder subtype card and add it to your hand, then bottom deck the rest in any order. 2

### S1-AZK01-072
- file: `S1-AZK01-072_Beanz-Mentor_E_R_die.jpg`
- card_code_guess: `AZK01-072`
- name_guess: `Beanz Mentor`
- type_guess: `ENTITY`
- subtype_guess: `Beanz, Elder`
- ocr_nums: `3, 2, 2`
- effect_guess: Once/Turn When Attacking If a Beanz entity in your Garden has 2 or more subtypes, it gets +1 attack until the end of the turn.

### S1-AZK01-073
- file: `S1-AZK01-073_Top-Beanz_E_C_die.jpg`
- card_code_guess: `AZK01-073`
- name_guess: `Top Beanz`
- type_guess: `ENTITY`
- subtype_guess: `Beanz`
- ocr_nums: `3, 2`
- effect_guess: When Attacking If you have 5 or more differently named entities in your Garden, this card gets +1 attack until the end of the turn.

### S1-AZK01-074
- file: `S1-AZK01-074_Gurugumi-Vanguard_E_UC_die.jpg`
- card_code_guess: `AZK01-074`
- name_guess: `Gurugumi Vanguard`
- type_guess: `ENTITY`
- subtype_guess: `Black Jade, Ninja`
- ocr_nums: `3, 2, 2`
- effect_guess: In Alley Only Ability Once/Turn Reveal 1 Black Jade entity in your hand: Play this card.

### S1-AZK01-075
- file: `S1-AZK01-075_Drunken-Brewmaster_E_C_die.jpg`
- card_code_guess: `AZK01-075`
- name_guess: `Drunken Brewmaster`
- type_guess: `ENTITY`
- subtype_guess: `Brewmaster, Elder`
- ocr_nums: `3, 2`
- effect_guess: On Play You may discard 1: Draw 1. 1

## Verified Batch: AZK01-067 to AZK01-075

- `AZK01-067` Frida
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `1`, gate `1`, attack `1`, health `1`
  - subtypes: `Beanz`, `Painter`
  - no rules text; flavor text only
- `AZK01-068` Pip
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `1`, gate `1`, attack `1`, health `1`
  - subtype: `Beanz`
  - effect: `[On Play] When this card is played in the Alley, draw 1, then discard 1.`
- `AZK01-069` Link
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `1`, gate `1`, attack `1`, health `1`
  - subtype: `Beanz`
  - effect: `[On Play] Look at the top 5 cards of your deck, reveal up to 1 Beanz subtype card and add it to your hand, then bottom deck the rest in any order.`
- `AZK01-070` Mocking Dummy
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `1`, gate `0`, attack `0`, health `2`
  - subtype: `Training Dummy`
  - effect: `[In Garden Only Ability][Response] Tap this card and deal 1 damage to it: Reduce an opponent entity's attack by 1 until the end of your opponent's turn.`
- `AZK01-071` Alley Fetchduck
  - `ENTITY`, `NORMAL`, rarity `UC`
  - cost `2`, gate `1`, attack `1`, health `2`
  - subtype: `Bird`
  - effect: `[End of Your Turn] You may bottom deck this card: Draw 1.`
- `AZK01-072` Beanz Mentor
  - `ENTITY`, `NORMAL`, rarity `R`
  - cost `2`, gate `1`, attack `1`, health `1`
  - subtypes: `Beanz`, `Elder`
  - effect: `[When Attacking] Give another Beanz subtype entity in your Garden +1 attack until the end of the turn.`
- `AZK01-073` Top Beanz
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `3`, gate `2`, attack `1`, health `1`
  - subtype: `Beanz`
  - effect: `If you only have Beanz subtype entities in your Garden, this card has +1 attack and +1 health.`
- `AZK01-074` Gurugumi Vanguard
  - `ENTITY`, `NORMAL`, rarity `UC`
  - cost `2`, gate `1`, attack `0`, health `2`
  - subtype: `Gurijutsu`
  - effect: `[Once/Turn][Main] This card's attack becomes the attack of an entity in your opponent's Garden until the end of the turn.`
- `AZK01-075` Drunken Brewmaster
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `4`, gate `1`, attack `2`, health `2`
  - subtypes: `Brewmaster`, `Elder`
  - effect: `[Once/Turn] You may sacrifice 2 Beanz subtype entities in your Garden: This card gets +2 attack and +2 health until the end of the turn.`

### S1-AZK01-077
- file: `S1-AZK01-077_Stalking-Assassin_E_C_die.jpg`
- card_code_guess: `AZK01-077`
- name_guess: `Stalking Assassin`
- type_guess: `ENTITY`
- subtype_guess: `Assassin, Ninja`
- ocr_nums: `3, 2`
- effect_guess: In Alley Only Ability Once/Turn Whenever another entity enters your opponent's Alley, you may play this card. 2

### S1-AZK01-078
- file: `S1-AZK01-078_Fermented-Beanz_E_C_die.jpg`
- card_code_guess: `AZK01-078`
- name_guess: `Fermented Beanz`
- type_guess: `ENTITY`
- subtype_guess: `Beanz`
- ocr_nums: `4, 2, 4`
- effect_guess: When Attacking If there are 4 or more differently named entities in your Garden, this card gets +2 attack until the end of the turn.

### S1-AZK01-080
- file: `S1-AZK01-080_Bladebound-Ally_E_R_die.jpg`
- card_code_guess: `AZK01-080`
- name_guess: `Bladebound Ally`
- type_guess: `ENTITY`
- subtype_guess: `Samurai, Weaponbound`
- ocr_nums: `4, 2, 2`
- effect_guess: When Equipped Give the equipped entity Charge (Can attack the same turn it enters the Garden).

### S1-AZK01-081
- file: `S1-AZK01-081_Gurugumi-Mentor_E_C_die.jpg`
- card_code_guess: `AZK01-081`
- name_guess: `Gurugumi Mentor`
- type_guess: `ENTITY`
- subtype_guess: `Black Jade, Elder`
- ocr_nums: `4, 2, 4`
- effect_guess: If this card has 5 or more attack, it has Infiltrate (When this card attacks, your opponent cannot activate cards with Defender). 4

### S1-AZK01-082
- file: `S1-AZK01-082_Black-Jade-Brawler_E_UC_die.jpg`
- card_code_guess: `AZK01-082`
- name_guess: `Black Jade Brawler`
- type_guess: `ENTITY`
- subtype_guess: `Black Jade, Weaponbound`
- ocr_nums: `4, 3, 2`
- effect_guess: Once/Turn When This Card Takes Damage If this card is in the Garden, it gets +1 attack until the end of the turn.

## Verified Batch: AZK01-077 to AZK01-082

- `AZK01-077` Stalking Assassin
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `3`, gate `2`, attack `2`, health `2`
  - subtypes: `Assassin`, `Steelborn`
  - effect: `This card can only attack leaders.`
- `AZK01-078` Fermented Beanz
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `4`, gate `2`, attack `2`, health `2`
  - subtype: `Beanz`
  - effect: `[When Destroyed] Deal 1 damage to a leader or an entity in your opponent's Garden.`
- `AZK01-080` Bladebound Ally
  - `ENTITY`, `NORMAL`, rarity `R`
  - cost `5`, gate `3`, attack `3`, health `2`
  - subtypes: `Invigorate`, `Samurai`
  - effect: `[On Play] If this card is played in the Garden, give another entity in your Garden +2 attack until the end of the turn.`
- `AZK01-081` Gurugumi Mentor
  - `ENTITY`, `NORMAL`, rarity `C`
  - cost `5`, gate `2`, attack `3`, health `3`
  - subtype: `Gurijutsu`
  - effect: `When this card is in your deck, it has all subtypes.`
- `AZK01-082` Black Jade Brawler
  - `ENTITY`, `NORMAL`, rarity `UC`
  - cost `5`, gate `2`, attack `3`, health `2`
  - keyword: `Charge`
  - subtypes: `Black Jade`, `Brawler`, `Steelborn`

### S1-AZK01-084
- file: `S1-AZK01-084_Good-Enough-Replica_S_C_die.jpg`
- card_code_guess: `AZK01-084`
- name_guess: `Good Enough Replica`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Main The next time up to 1 entity in your Garden would be returned to hand this turn, it stays in play instead.

### S1-AZK01-085
- file: `S1-AZK01-085_Invigorating-Concoction_S_C_die.jpg`
- card_code_guess: `AZK01-085`
- name_guess: `Invigorating Concoction`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: [Main] Give an entity in your Garden +1/+1 until the end of the turn.

### S1-AZK01-086
- file: `S1-AZK01-086_Forging-Tricks_S_UC_die.jpg`
- card_code_guess: `AZK01-086`
- name_guess: `Forging Tricks`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: Main Return a Weapon in your Garden to your hand: An entity in your Garden gets +2/+2 until the end of the tum.

### S1-AZK01-087
- file: `S1-AZK01-087_Mizuryuus-Torrent_S_SR_die.jpg`
- card_code_guess: `AZK01-087`
- name_guess: `Mizuryuus Torrent`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: Main If your leader is Wavecaller and has 12 or less health, draw 3. 3

### S1-AZK01-088
- file: `S1-AZK01-088_Pulled-Under_S_R_die.jpg`
- card_code_guess: `AZK01-088`
- name_guess: `Pulled Under`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `3`
- effect_guess: Main If your leader is Wavecaller, freeze up to 3 entities in your opponent's Garden until the start of your next turn. 3

### S1-AZK01-089
- file: `S1-AZK01-089_Mizuryuu-Fist-Master_E_UC_die.jpg`
- card_code_guess: `AZK01-089`
- name_guess: `Mizuryuu Fist Master`
- type_guess: `ENTITY`
- subtype_guess: `Monk, Wavecaller`
- ocr_nums: `2, 2`
- effect_guess: Once/Turn In Garden Only Ability Return 1 Watercrafting spell in your Garden to your hand: This card gets +2 attack until the end of the turn.

## Verified Batch: AZK01-084 to AZK01-089

- `AZK01-084` Good Enough Replica
  - `SPELL`, `NORMAL`, rarity `C`
  - cost `2`
  - subtype: `Trickster`
  - effect: `[Main] Return a Normal entity card with a cost of 6 or less from your discard pile to your hand.`
- `AZK01-085` Invigorating Concoction
  - `SPELL`, `NORMAL`, rarity `C`
  - cost `3`
  - subtypes: `Beanz`, `Invigorate`
  - effect: `[Main][Response] Give an entity in your Garden +2 attack until the end of the turn.`
- `AZK01-086` Forging Tricks
  - `SPELL`, `NORMAL`, rarity `UC`
  - cost `4`
  - subtypes: `Steelborn`, `Trickster`
  - effect: `[Main] You may place up to 5 Weapon cards from your discard pile to the bottom of your deck in any order: Until the end of the turn, give your leader +1 attack for each card you bottom decked.`
- `AZK01-087` Mizuryuu's Torrent
  - `SPELL`, `WATER`, rarity `SR`
  - cost `4`
  - subtype: `Mizuryuu`
  - effect: `[Main] Put up to 2 entities with a combined cost of 5 or less in your opponent's Garden to the bottom of their owner's deck in any order.`
- `AZK01-088` Pulled Under
  - `SPELL`, `WATER`, rarity `R`
  - cost `6`
  - subtype: `Watercrafting`
  - effect: `[Main] Put an entity with a cost of 7 or less in your opponent's Garden to the bottom of its owner's deck.`
- `AZK01-089` Mizuryuu Fist Master
  - `ENTITY`, `WATER`, rarity `UC`
  - cost `7`, gate `3`, attack `5`, health `5`
  - subtypes: `Frog`, `Mizuryuu`, `Watercrafting`
  - effect: `[In Alley Only Ability][Main] You may bottom deck this card: Put up to 2 entity cards with a combined cost of 5 or less in your opponent's Garden to the bottom of their owner's deck in any order.`

### S1-AZK01-090
- file: `S1-AZK01-090_Priestess-of-the-Mists_E_C_die.jpg`
- card_code_guess: `AZK01-090`
- name_guess: `Priestess of the Mists`
- type_guess: `ENTITY`
- subtype_guess: `Elder, Wavecaller`
- ocr_nums: `1, 2`
- effect_guess: On Play If your leader is Wavecaller, draw 1.

### S1-AZK01-091
- file: `S1-AZK01-091_Bubble-Adept_E_UC_die.jpg`
- card_code_guess: `AZK01-091`
- name_guess: `Bubble Adept`
- type_guess: `ENTITY`
- subtype_guess: `Painter, Watercrafting`
- ocr_nums: `3, 2`
- effect_guess: Once/Turn Main Return 1 of your Mizuryuu entities to your hand: Play 1 entity with a cost of 3 or less from your hand. 3

### S1-AZK01-092
- file: `S1-AZK01-092_Lotus-of-Reflection_S_C_die.jpg`
- card_code_guess: `AZK01-092`
- name_guess: `Lotus of Reflection`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Response If you would take damage this turn, instead you may return 1 entity in your Garden to your hand. 1

### S1-AZK01-093
- file: `S1-AZK01-093_Naiyara-the-Tideweaver_E_R_die.jpg`
- card_code_guess: `AZK01-093`
- name_guess: `Naiyara the Tideweaver`
- type_guess: `ENTITY`
- subtype_guess: `Elder, Watercrafting`
- ocr_nums: `4, 2, 3`
- effect_guess: Once/Turn Whenever an entity enters your Garden by a card effect, it gets +1/+1 until the end of the turn.

### S1-AZK01-094
- file: `S1-AZK01-094_Hidden-Dagger_W_C_die.jpg`
- card_code_guess: `AZK01-094`
- name_guess: `Hidden Dagger`
- type_guess: `WEAPON`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: On Play If you have a Shockcoil card in your Garden or Alley, deal 1 damage to an entity or a leader in any Garden.

### S1-AZK01-095
- file: `S1-AZK01-095_Stormglass-Katana_W_C_die.jpg`
- card_code_guess: `AZK01-095`
- name_guess: `Stormglass Katana`
- type_guess: `WEAPON`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: When Attacking If the attacking entity is a Ninja, it becomes Shocked after the attack. 1

### S1-AZK01-096
- file: `S1-AZK01-096_Ninpo-Thunderstep_S_UC_die.jpg`
- card_code_guess: `AZK01-096`
- name_guess: `Ninpo Thunderstep`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Response Untap up to 1 Ninja entity in your Garden.

### S1-AZK01-097
- file: `S1-AZK01-097_Black-Jade-Pawnbroker_E_C_die.jpg`
- card_code_guess: `AZK01-097`
- name_guess: `Black Jade Pawnbroker`
- type_guess: `ENTITY`
- subtype_guess: `Black Jade, Merchant`
- ocr_nums: `1, 1`
- effect_guess: When Returned to Hand You may discard 1: Draw 1.

### S1-AZK01-098
- file: `S1-AZK01-098_Arms-Dealer-Kin_E_C_die.jpg`
- card_code_guess: `AZK01-098`
- name_guess: `Arms Dealer Kin`
- type_guess: `ENTITY`
- subtype_guess: `Merchant, Monkey`
- ocr_nums: `2, 1`
- effect_guess: When Attacking or When This Card is Returned to Hand If you have a Weapon in your Garden, draw 1, then discard 1. 1

### S1-AZK01-100
- file: `S1-AZK01-100_Raizans-Riposte_S_C_die.jpg`
- card_code_guess: `AZK01-100`
- name_guess: `Raizans Riposte`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: Response If your leader is Raizan, it gets +2 attack until the end of the turn.

### S1-AZK01-101
- file: `S1-AZK01-101_Sand-Stands-Still_S_SR_die.jpg`
- card_code_guess: `AZK01-101`
- name_guess: `Sand Stands Still`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `4`
- effect_guess: Main If your leader is Earth and has 12 or less health, draw 3.

### S1-AZK01-102
- file: `S1-AZK01-102_Oathstone_S_C_die.jpg`
- card_code_guess: `AZK01-102`
- name_guess: `Oathstone`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Response Your leader gains +2 health until the end of the turn.

### S1-AZK01-103
- file: `S1-AZK01-103_Dropline-Station_E_UC_die.jpg`
- card_code_guess: `AZK01-103`
- name_guess: `Dropline Station`
- type_guess: `ENTITY`
- subtype_guess: `Outpost`
- ocr_nums: `2, 0, 3`
- effect_guess: End of Turn If there are 2 or more entities in your opponent's Garden, you may play 1 entity with a cost of 1 or less from your hand. 1

### S1-AZK01-104
- file: `S1-AZK01-104_Sanzus-Envoy_E_UC_die.jpg`
- card_code_guess: `AZK01-104`
- name_guess: `Sanzus Envoy`
- type_guess: `ENTITY`
- subtype_guess: `Caravan, Scavenger`
- ocr_nums: `3, 1, 2`
- effect_guess: When This Card is Returned to Hand or Destroyed Draw 1.

### S1-AZK01-105
- file: `S1-AZK01-105_Prickly-Tumbleweed_E_C_die.jpg`
- card_code_guess: `AZK01-105`
- name_guess: `Prickly Tumbleweed`
- type_guess: `ENTITY`
- subtype_guess: `Defender, Plant`
- ocr_nums: `1, 3`
- effect_guess: When This Card is Returned to Hand or Destroyed Deal 2 damage to a leader or entity in any Garden.

### S1-AZK01-106
- file: `S1-AZK01-106_Lord-of-Sands-Osunanami_E_SR_die.jpg`
- card_code_guess: `AZK01-106`
- name_guess: `Lord of Sands Osunanami`
- type_guess: `ENTITY`
- subtype_guess: `Caravan, Elder`
- ocr_nums: `4, 4, 4`
- effect_guess: Once/Turn Whenever another entity enters your Garden, you may return that entity to your hand. If you do, draw 1.

### S1-AZK01-107
- file: `S1-AZK01-107_Offering-to-Stillstone_S_C_die.jpg`
- card_code_guess: `AZK01-107`
- name_guess: `Offering to Stillstone`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Main Return 1 entity in your Garden to your hand: Heal 2 to your leader. 2

### S1-AZK01-108
- file: `S1-AZK01-108_Crushing-Weight_S_R_die.jpg`
- card_code_guess: `AZK01-108`
- name_guess: `Crushing Weight`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: Response Return 1 entity in your Garden to your hand: Deal 2 damage to a leader or entity in any Garden.

### S1-AZK01-109
- file: `S1-AZK01-109_Rock-Sloth_E_C_die.jpg`
- card_code_guess: `AZK01-109`
- name_guess: `Rock Sloth`
- type_guess: `ENTITY`
- subtype_guess: `Scavenger, Sloth`
- ocr_nums: `1, 4`
- effect_guess: Carapace 1 (Reduce damage from all sources by 1. Carapace stacks.)

### S1-AZK01-110
- file: `S1-AZK01-110_Gluttonous-Devourer-Kasha_E_UC_die.jpg`
- card_code_guess: `AZK01-110`
- name_guess: `Gluttonous Devourer Kasha`
- type_guess: `ENTITY`
- subtype_guess: `Brawler, Cultivator, Yonkai`
- ocr_nums: `4, 2, 4`
- effect_guess: Once/Turn When Attacking Return 1 Earth entity in your Garden to your hand: This card gains +X attack until the end of the turn, where X equals the returned entity's health. 4

### S1-AZK01-111
- file: `S1-AZK01-111_Black-Jade-Decoy_E_R_die.jpg`
- card_code_guess: `AZK01-111`
- name_guess: `Black Jade Decoy`
- type_guess: `ENTITY`
- subtype_guess: `Black Jade, Dummy`
- ocr_nums: `5, 2, 3`
- effect_guess: In Alley Only Ability Whenever a card in your Garden would be returned to your hand, you may destroy this card instead. 3

### S1-AZK01-112A
- file: `S1-AZK01-112A_Enrai-Shakunetsu_E_SR_die.jpg`
- card_code_guess: `AZK01-112A`
- name_guess: `Enrai Shakunetsu`
- type_guess: `UNKNOWN`
- subtype_guess: `Black Jade, Blazeartisan, Elder`
- ocr_nums: `4, 4`
- effect_guess: Once/Turn When This Card Takes Damage You must deal 1 damage to your leader: Put a +1/+1 counter on this card. 4

### S1-AZK01-112
- file: `S1-AZK01-112_Enrai-Shakunetsu_E_SR_die.jpg`
- card_code_guess: `AZK01-112`
- name_guess: `Enrai Shakunetsu`
- type_guess: `ENTITY`
- subtype_guess: `Black Jade, Blazeartisan, Elder`
- ocr_nums: `4, 4`
- effect_guess: Once/Turn When This Card Takes Damage You must deal 1 damage to your leader: Put a +1/+1 counter on this card. 4

### S1-AZK01-113
- file: `S1-AZK01-113_Cinderwake-Pursuer_E_C_die.jpg`
- card_code_guess: `AZK01-113`
- name_guess: `Cinderwake Pursuer`
- type_guess: `ENTITY`
- subtype_guess: `Cinderwake, Pyreskin`
- ocr_nums: `1, 1`
- effect_guess: On Play If you have another Pyreskin in your Garden, draw 1.

### S1-AZK01-114
- file: `S1-AZK01-114_Omen-Peddler_E_R_die.jpg`
- card_code_guess: `AZK01-114`
- name_guess: `Omen Peddler`
- type_guess: `ENTITY`
- subtype_guess: `Crimsondrift, Siren`
- ocr_nums: `2, 1`
- effect_guess: In Garden Only Ability Once/Turn While your health is 10 or less, deal 1 damage to your leader: This card gains +2 attack until the end of the turn. 3

### S1-AZK01-115
- file: `S1-AZK01-115_Crazed-Arsonist_E_C_die.jpg`
- card_code_guess: `AZK01-115`
- name_guess: `Crazed Arsonist`
- type_guess: `ENTITY`
- subtype_guess: `Blazeartisan, Black Jade`
- ocr_nums: `2, 1`
- effect_guess: When This Card is Returned to Hand or Destroyed Deal 1 damage to your leader. Deal 1 damage to a leader or entity in any Garden.

### S1-AZK01-116
- file: `S1-AZK01-116_Tenmoku-Daiki_E_R_die.jpg`
- card_code_guess: `AZK01-116`
- name_guess: `Tenmoku Daiki`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Elder, Pyreskin`
- ocr_nums: `4, 3, 3`
- effect_guess: Once/Turn When This Card Takes Damage You may play 1 entity with a cost of 3 or less from your hand. 3

### S1-AZK01-117
- file: `S1-AZK01-117_Ignition-Pact_S_UC_die.jpg`
- card_code_guess: `AZK01-117`
- name_guess: `Ignition Pact`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Main Deal 1 damage to an entity in your Garden: Draw 2, then discard 1.

### S1-AZK01-118
- file: `S1-AZK01-118_Bandit-Ringleader_E_C_die.jpg`
- card_code_guess: `AZK01-118`
- name_guess: `Bandit Ringleader`
- type_guess: `ENTITY`
- subtype_guess: `Bandit, Pyreskin`
- ocr_nums: `2, 2`
- effect_guess: When Attacking If this card's attack is 4 or more, it gains Infiltrate until the end of the turn.

### S1-AZK01-119A
- file: `S1-AZK01-119A_Piko-of-Thousand-Blades_L_L_die.jpg`
- card_code_guess: `AZK01-119A`
- name_guess: `Piko of Thousand Blades`
- type_guess: `CADER`
- subtype_guess: `Piko, Steelborn`
- ocr_nums: `20`
- effect_guess: Once/Turn Main Pay 1 IKZ: Until the end of the turn, the first time an entity in your Garden or Alley becomes Shocked, deal 1 damage to a leader or entity in any Garden.

### S1-AZK01-120
- file: `S1-AZK01-120_Stormchain-Gate_G_G_die.jpg`
- card_code_guess: `AZK01-120`
- name_guess: `Stormchain Gate`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: On Gate Portal If the portaled entity has the Shockcoil or Steelborn subtype, deal 1 damage to a leader or entity in any Garden.

### S1-AZK01-121A
- file: `S1-AZK01-121A_Kagoro-of-the-Burnt-Path_L_L_die.jpg`
- card_code_guess: `AZK01-121A`
- name_guess: `Kagoro of the Burnt Path`
- type_guess: `CADER`
- subtype_guess: `Kagoro, Pyreskin`
- ocr_nums: `20`
- effect_guess: Once/Turn Main Pay 1 IKZ: Until the end of the turn, the first time an entity in your Garden or Alley is damaged, it gets +2 attack.

### S1-AZK01-122
- file: `S1-AZK01-122_Rushfire-Gate_G_G_die.jpg`
- card_code_guess: `AZK01-122`
- name_guess: `Rushfire Gate`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: On Gate Portal If the portaled entity has the Pyreskin or Blazeartisan subtype, deal 1 damage to your leader: That entity gets +1 attack until the end of the turn.

### S1-AZK01-123A
- file: `S1-AZK01-123A_Goro-Graveloth_L_L_die.jpg`
- card_code_guess: `AZK01-123A`
- name_guess: `Goro Graveloth`
- type_guess: `CADER`
- subtype_guess: `Caravan, Goro`
- ocr_nums: `20`
- effect_guess: Once/Turn Main Pay 1 IKZ: Until the end of the turn, the first time an entity enters your Garden by a card effect, draw 1.

### S1-AZK01-124
- file: `S1-AZK01-124_Gate-of-Devotion-Gate_G_G_die.jpg`
- card_code_guess: `AZK01-124`
- name_guess: `Gate of Devotion Gate`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: On Gate Portal If the portaled entity has the Scavenger or Caravan subtype, heal 1 to your leader. 1

### S1-AZK01-125A
- file: `S1-AZK01-125A_Benzai-the-Sly_L_L_die.jpg`
- card_code_guess: `AZK01-125A`
- name_guess: `Benzai the Sly`
- type_guess: `CADER`
- subtype_guess: `Benzai, Wavecaller`
- ocr_nums: `20`
- effect_guess: Once/Turn Main Pay 1 IKZ: Until the end of the turn, the first time an entity enters your Garden by a card effect, it gets +1 health.

### S1-AZK01-126
- file: `S1-AZK01-126_Gate-of-Echoed-Waves-Gate_G_G_die.jpg`
- card_code_guess: `AZK01-126`
- name_guess: `Gate of Echoed Waves Gate`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: On Gate Portal If the portaled entity has the Watercrafting or Driftward subtype, you may retum 1 entity in your Garden to your hand. If you do, draw 1.

### S1-AZK01-127
- file: `S1-AZK01-127_Sundering-Strike_S_UC_die.jpg`
- card_code_guess: `AZK01-127`
- name_guess: `Sundering Strike`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `3`
- effect_guess: Main Deal 1 damage to your leader: Deal up to 3 damage to an entity in your opponent's Garden.

### S1-AZK01-128
- file: `S1-AZK01-128_Wrong-Step_S_UC_die.jpg`
- card_code_guess: `AZK01-128`
- name_guess: `Wrong Step`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: Response If there is no entity in your Garden with Defender, the attacking entity gets -3 attack until the end of the tum.

### S1-AZK01-129
- file: `S1-AZK01-129_Silk-Tongue-Velya_E_UC_die.jpg`
- card_code_guess: `AZK01-129`
- name_guess: `Silk Tongue Velya`
- type_guess: `ENTITY`
- subtype_guess: `Companion, Elder, Trickster`
- ocr_nums: `3, 1, 3`
- effect_guess: On Play Choose 1 entity in your Garden. It gains 1 subtype of your choice until the end of the turn.

## STT03

### S1-STT03-001A
- file: `S1-STT03-001A_Bobu_L_L_die.jpg`
- card_code_guess: `STT03-001A`
- name_guess: `Bobu`
- type_guess: `CADER`
- subtype_guess: `Bobu, Brewmaster, Stonemend`
- ocr_nums: `20`
- effect_guess: Once/Turn Main * IKZ: Until the start of your next turn, the first time an A entity in your Garden or Alley is destroyed or sacrificed, you may heal 1 to your leader.

### S1-STT03-002A
- file: `S1-STT03-002A_Stonehaven-Gate_G_G_die.jpg`
- card_code_guess: `STT03-002A`
- name_guess: `Stonehaven Gate`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: On Gate Portal If the portaled entity has the Cultivator or Yonkai subtype, you may return 1 entity in your Garden to your hand. If you do, heal 1 to your leader. 1

### S1-STT03-003
- file: `S1-STT03-003_Koyama-Farm-Potter_E_C_die.jpg`
- card_code_guess: `STT03-003`
- name_guess: `Koyama Farm Potter`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Elder, Stonemend`
- ocr_nums: `1, 2`
- effect_guess: On Play Look at the top 4 cards of your deck, reveal up to 1 Cultivator, Plant, or Sloth card and add it to your hand, then bottom deck the rest in any order.

### S1-STT03-004
- file: `S1-STT03-004_Sloth-Scarecrow_E_C_die.jpg`
- card_code_guess: `STT03-004`
- name_guess: `Sloth Scarecrow`
- type_guess: `ENTITY`
- subtype_guess: `Dummy, Sloth, Stonemend`
- ocr_nums: `2, 2`
- effect_guess: Once/Turn Main If you played an entity into your Garden this tum, you may destroy this card: Until the start of your next tum, that entity cannot be damaged by card effects.

### S1-STT03-005
- file: `S1-STT03-005_Wobbly-Cabbage-Cart_E_C_die.jpg`
- card_code_guess: `STT03-005`
- name_guess: `Wobbly Cabbage Cart`
- type_guess: `ENTITY`
- subtype_guess: `Cart, Cultivator, Plant`
- ocr_nums: `2, 1, 2`
- effect_guess: In Alley Only Ability Whenever another entity enters your Garden, you may return 1 Plant, Sloth, or Stonemend card from your discard pile to your hand. 1

### S1-STT03-006
- file: `S1-STT03-006_Cactus-Farmer_E_UC_die.jpg`
- card_code_guess: `STT03-006`
- name_guess: `Cactus Farmer`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Elder, Yonkai`
- ocr_nums: `3, 1, 2`
- effect_guess: When This Card is Returned to Hand or Destroyed You may play 1 Cultivator, Plant, or Sloth card with a cost of 1 or less from your hand. 1

### S1-STT03-007
- file: `S1-STT03-007_Koyama-Farm-Caretaker_E_R_die.jpg`
- card_code_guess: `STT03-007`
- name_guess: `Koyama Farm Caretaker`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Elder, Yonkai`
- ocr_nums: `3, 1, 2`
- effect_guess: Once/Turn End of Turn Return another Cultivator, Plant, or Sloth card in your Garden to your hand. If you do, draw 1. 1

### S1-STT03-008
- file: `S1-STT03-008_Midnight-Courier_E_C_die.jpg`
- card_code_guess: `STT03-008`
- name_guess: `Midnight Courier`
- type_guess: `ENTITY`
- subtype_guess: `Bird, Scavenger, Stonemend`
- ocr_nums: `3, 1, 2`
- effect_guess: When Attacking If you have 5 or more cards in hand, this card gets +1 attack until the end of the tum.

### S1-STT03-009
- file: `S1-STT03-009_Warding-Totem_E_UC_die.jpg`
- card_code_guess: `STT03-009`
- name_guess: `Warding Totem`
- type_guess: `ENTITY`
- subtype_guess: `Defender, Totem`
- ocr_nums: `2, 2`
- effect_guess: Whenever a Sloth or Yonkai entity enters your Garden, it gains Carapace 1 until the end of the tum. 1

### S1-STT03-010
- file: `S1-STT03-010_Shroommancer_E_C_die.jpg`
- card_code_guess: `STT03-010`
- name_guess: `Shroommancer`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Plant, Stonemend`
- ocr_nums: `3, 1, 2`
- effect_guess: On Play You may discard 1: Give an entity in your Garden +1/+1 until the end of the turn.

### S1-STT03-011
- file: `S1-STT03-011_Koyama-Farm-Plowman_E_C_die.jpg`
- card_code_guess: `STT03-011`
- name_guess: `Koyama Farm Plowman`
- type_guess: `ENTITY`
- subtype_guess: `Companion, Cultivator, Stonemend`
- ocr_nums: `3, 2`
- effect_guess: In Alley Only Ability Once/Turn Whenever one or more entities in your Garden are returned to your hand, you may play this card.

### S1-STT03-012
- file: `S1-STT03-012_Miharu-of-the-White-Bloom_E_SR_die.jpg`
- card_code_guess: `STT03-012`
- name_guess: `Miharu of the White Bloom`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Elder, Stonemend`
- ocr_nums: `3, 3`
- effect_guess: Once/Turn When This Card is Returned to Hand or Destroyed You may play up to 2 Cultivator, Plant, or Sloth cards with a total cost of 3 or less from your hand. 3

### S1-STT03-013A
- file: `S1-STT03-013A_Stone-Masked-Ancient_E_SR_die.jpg`
- card_code_guess: `STT03-013A`
- name_guess: `Stone Masked Ancient`
- type_guess: `UNKNOWN`
- subtype_guess: `Elder, Sloth, Yonkai`
- ocr_nums: `4, 4`
- effect_guess: In Garden Only Ability The first time each turn that another entity would be returned to your hand, you may heal 1 to your leader instead.

### S1-STT03-014
- file: `S1-STT03-014_Sandcoil-Python_E_UC_die.jpg`
- card_code_guess: `STT03-014`
- name_guess: `Sandcoil Python`
- type_guess: `ENTITY`
- subtype_guess: `Caravan, Serpent, Stonemend`
- ocr_nums: `2, 2`
- effect_guess: Once/Turn In Garden Only Ability Return another entity in your Garden to your hand: This card gets +2 attack until the end of the turn.

### S1-STT03-015
- file: `S1-STT03-015_Jar-of-Beans_S_UC_die.jpg`
- card_code_guess: `STT03-015`
- name_guess: `Jar of Beans`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Main Play up to 2 entities with a total cost of 2 or less from your hand.

### S1-STT03-016
- file: `S1-STT03-016_Quicksand_S_R_die.jpg`
- card_code_guess: `STT03-016`
- name_guess: `Quicksand`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `3`
- effect_guess: Response Return 1 entity in your Garden to your hand: Deal 2 damage to each entity in any Garden.

## STT04

### S1-STT04-001
- file: `S1-STT04-001_Zero_L_L_die.jpg`
- card_code_guess: `STT04-001`
- name_guess: `Zero`
- type_guess: `CADER`
- subtype_guess: `Pyreskin, Zero`
- ocr_nums: `20`
- effect_guess: Once/Turn Main Pay 1 IKZ: Until the start of your next turn, the first time an entity in your Garden or Alley is damaged by a card effect, draw 1.

### S1-STT04-002A
- file: `S1-STT04-002A_Ragefire-Gate_G_G_die.jpg`
- card_code_guess: `STT04-002A`
- name_guess: `Ragefire Gate`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: ``
- effect_guess: On Gate Portal If the portaled entity has the Pyreskin or Cinderwake subtype, deal 1 damage to your leader: If you do, draw 1.

### S1-STT04-003
- file: `S1-STT04-003_Cinderwake-Seer_E_UC_die.jpg`
- card_code_guess: `STT04-003`
- name_guess: `Cinderwake Seer`
- type_guess: `ENTITY`
- subtype_guess: `Cinderwake, Elder, Pyreskin`
- ocr_nums: `1, 2`
- effect_guess: On Play Look at the top 4 cards of your deck, reveal up to 1 Pyreskin or Cinderwake card and add it to your hand, then bottom deck the rest in any order.

### S1-STT04-004
- file: `S1-STT04-004_Fanatic-Kindler_E_C_die.jpg`
- card_code_guess: `STT04-004`
- name_guess: `Fanatic Kindler`
- type_guess: `ENTITY`
- subtype_guess: `Cultivator, Pyreskin`
- ocr_nums: `2, 1, 2`
- effect_guess: In Alley Only Ability Whenever another entity in your Garden is dealt damage by a card effect, you may heal 1 to your leader.

### S1-STT04-005
- file: `S1-STT04-005_Ruby_E_C_die.jpg`
- card_code_guess: `STT04-005`
- name_guess: `Ruby`
- type_guess: `ENTITY`
- subtype_guess: `Bird, Pyreskin`
- ocr_nums: `2, 1, 2`
- effect_guess: Once/Turn When Attacking If your leader is a Pyreskin, this card gets +1 attack until the end of the tum.

### S1-STT04-006
- file: `S1-STT04-006_Wolf-Cub_E_C_die.jpg`
- card_code_guess: `STT04-006`
- name_guess: `Wolf Cub`
- type_guess: `ENTITY`
- subtype_guess: `Beast, Pyreskin`
- ocr_nums: `2, 1, 2`
- effect_guess: Once/Turn When Attacking If your leader is a Pyreskin, this card gets +1 attack until the end of the tum.

### S1-STT04-007
- file: `S1-STT04-007_Enraged-Howler_E_C_die.jpg`
- card_code_guess: `STT04-007`
- name_guess: `Enraged Howler`
- type_guess: `ENTITY`
- subtype_guess: `Pyreskin, Wolf`
- ocr_nums: `3, 1, 2`
- effect_guess: Once/Turn When This Card Takes Damage This card gains +2 attack until the end of the turn.

### S1-STT04-008
- file: `S1-STT04-008_Lady-Emberheart_E_UC_die.jpg`
- card_code_guess: `STT04-008`
- name_guess: `Lady Emberheart`
- type_guess: `ENTITY`
- subtype_guess: `Crimsondrift`
- ocr_nums: `3, 2`
- effect_guess: Once/Turn On your turn, after this card attacks an entity, you may untap this card.

### S1-STT04-009
- file: `S1-STT04-009_Cinderwake-Ritualist_E_R_die.jpg`
- card_code_guess: `STT04-009`
- name_guess: `Cinderwake Ritualist`
- type_guess: `ENTITY`
- subtype_guess: `Cinderwake, Pyreskin`
- ocr_nums: `3, 21, 2`
- effect_guess: In Garden Only Ability Once/Turn Whenever this card takes damage from card effects, you may also deal that damage to another leader or entity in any Garden, capped at 2 damage.

### S1-STT04-010
- file: `S1-STT04-010_Reckless-Tinkerer_E_C_die.jpg`
- card_code_guess: `STT04-010`
- name_guess: `Reckless Tinkerer`
- type_guess: `ENTITY`
- subtype_guess: `Blazerker, Elder, Pyreskin`
- ocr_nums: `3, 2`
- effect_guess: Charge (Can attack the same turn it enters the Garden) On Play You must deal 1 damage to this card.

### S1-STT04-011
- file: `S1-STT04-011_Scorchland-Raven_E_C_die.jpg`
- card_code_guess: `STT04-011`
- name_guess: `Scorchland Raven`
- type_guess: `ENTITY`
- subtype_guess: `Bird`
- ocr_nums: `4, 13`
- effect_guess:

### S1-STT04-012
- file: `S1-STT04-012_Spiteful-Raider_E_UC_die.jpg`
- card_code_guess: `STT04-012`
- name_guess: `Spiteful Raider`
- type_guess: `ENTITY`
- subtype_guess: `Bandit, Pyreskin`
- ocr_nums: `4, 2`
- effect_guess: Once/Turn Whenever this card takes damage, deal 1 damage to a leader or an entity in any Garden.

### S1-STT04-013
- file: `S1-STT04-013_Kurai-the-Volcano_E_SR_die.jpg`
- card_code_guess: `STT04-013`
- name_guess: `Kurai the Voleano`
- type_guess: `ENTITY`
- subtype_guess: `Crimsondrift, Elder`
- ocr_nums: `5, 3, 3`
- effect_guess: Once/Turn Whenever an entity in your opponentis Garden is destroyed, untap this card. 3

### S1-STT04-014A
- file: `S1-STT04-014A_Scorchveil-Shinobi-Suzuka_E_SR_die.jpg`
- card_code_guess: `STT04-014A`
- name_guess: `Scorchveil Shinobi, Suzuka`
- type_guess: `UNKNOWN`
- subtype_guess: `Blazerker, Ninja`
- ocr_nums: `3, 3`
- effect_guess: Charge (Can attack the same tum it enters the Garden) On Play liyour leader has the Scorchweaver subtype, deal 1 damage to all entities in each player's 3

### S1-STT04-015
- file: `S1-STT04-015_Detonation-Pact_S_C_die.jpg`
- card_code_guess: `STT04-015`
- name_guess: `Detonation Pact`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `1`
- effect_guess: Main Deal 1 damage to your leader: Deal 2 damage to your opponent's leader.

### S1-STT04-016
- file: `S1-STT04-016_Collateral-Burst_S_UC_die.jpg`
- card_code_guess: `STT04-016`
- name_guess: `Collateral Burst`
- type_guess: `UNKNOWN`
- subtype_guess: ``
- ocr_nums: `2`
- effect_guess: Main Deal 1 damage to an entity in your Garden: Deal up to 2 damage to a leader or an entity in your opponent's Garden.

### S1-STT04-017
- file: `S1-STT04-017_Wrath-of-Sinder_S_R_die.jpg`
- card_code_guess: `STT04-017`
- name_guess: `Wrath of Sinder`
- type_guess: `SPELL`
- subtype_guess: ``
- ocr_nums: `3`
- effect_guess: Main Sacrifice any number of entities in your Garden: Deal damage to an entity or a leader in any Garden equal to the number of entities sacrificed.

## Verified 2026-03-20: AZK01-090 to AZK01-093

- `AZK01-090` Priestess of the Mists
  - `ENTITY`, `WATER`, rarity `C`
  - cost `5`, gate `2`, atk `3`, hp `3`
  - subtypes: `Priestess`, `Wavecaller`
  - effect: `[On Play] You may return an entity with a cost of 2 or less in your Garden to your hand.`
- `AZK01-091` Bubble Adept
  - `ENTITY`, `WATER`, rarity `UC`
  - cost `2`, gate `1`, atk `2`, hp `1`
  - subtypes: `Aquashield`, `Driftward`
  - effect: `[In Garden Only Ability][Response] You may sacrifice this card: Reduce an entity's attack by 1 until the end of your opponent's turn.`
- `AZK01-092` Lotus of Reflection
  - `SPELL`, `WATER`, rarity `C`
  - cost `2`
  - subtypes: `Lotus`, `Wavecaller`
  - effect: `[Main] Look at the top 5 cards of your deck, you may reveal a Water card with a cost of 2 or less and add it to your hand, then bottom deck the rest in any order. You may play the card from your hand.`
- `AZK01-093` Naiyara the Tideweaver
  - `ENTITY`, `WATER`, rarity `R`
  - cost `7`, gate `3`, atk `4`, hp `4`
  - subtypes: `Downcurrent`, `Mizuryuu`
  - effect: `[On Play] If this card is played in the Garden, put an entity with a cost of 4 or less in your opponent's Garden to the bottom of its owner's deck.`

## Verified 2026-03-20: AZK01-094 to AZK01-103

- `AZK01-094` Hidden Dagger
  - `WEAPON`, `LIGHTNING`, rarity `C`
  - cost `1`, atk `1`
  - subtypes: `Dagger`, `Shadowfang`
  - effect: `This card can be played as a Response.`
- `AZK01-095` Stormglass Katana
  - `WEAPON`, `LIGHTNING`, rarity `C`
  - cost `4`, atk `3`
  - subtypes: `Riftwalk`, `Sword`
  - effect: `When equipped to a leader, the leader can attack untapped or tapped entities in your opponent's Alley.`
- `AZK01-096` Ninpo: Thunderstep
  - `SPELL`, `LIGHTNING`, rarity `UC`
  - cost `2`
  - subtypes: `Voltguard`
  - effect: `[Response] Swap an entity in your Garden with an entity in your Alley. If the other entity being swapped out is the target of an attack, the Alley entity being swapped into the Garden becomes the new target.`
- `AZK01-097` Black Jade Pawnbroker
  - `ENTITY`, `LIGHTNING`, rarity `C`
  - cost `2`, gate `1`, atk `1`, hp `1`
  - subtypes: `Black Jade`, `Merchant`, `Steelborn`
  - effect: `[On Play] Put 5 cards from the top of your deck into your discard pile. If any of the discarded cards are Weapon cards, you may add 1 of them to your hand.`
- `AZK01-098` Arms Dealer, Kin
  - `ENTITY`, `LIGHTNING`, rarity `C`
  - cost `4`, gate `1`, atk `2`, hp `2`
  - subtypes: `Merchant`, `Steelborn`
  - effect: `[On Play] If this card is played in the Alley you may tap this card: Play a Weapon card with a cost of 3 or less from your hand.`
- `AZK01-100` Raizan's Riposte
  - `SPELL`, `LIGHTNING`, rarity `C`
  - cost `2`
  - subtypes: `Raizan`, `Steelborn`
  - effect: `[Response] Play from your discard pile a Weapon card with a cost of 2 or less.`
- `AZK01-101` Sand Stands Still
  - `SPELL`, `EARTH`, rarity `SR`
  - cost `3`
  - subtypes: `Earthwarden`
  - effect: `[Main][Response] Give an Earth entity in your Garden +3 health until the end of turn.`
- `AZK01-102` Oathstone
  - `SPELL`, `EARTH`, rarity `C`
  - cost `2`
  - subtypes: `Obsidian`
  - effect: `[Response] Give an entity with a cost of 4 or less Carapace 1 until the end of turn.`
- `AZK01-103` Dropline Station
  - `ENTITY`, `EARTH`, rarity `UC`
  - cost `5`, gate `3`, atk `0`, hp `4`
  - subtypes: `Earthfury`
  - effect: `[In Garden Only Ability] Sacrifice an untapped Earth entity in your Garden: Deal damage equal to the sacrificed card's health to a leader, capped at 5 damage. If the sacrificed entity had 3 or more health, draw 1. (This ability is not affected by Cooldown)`
