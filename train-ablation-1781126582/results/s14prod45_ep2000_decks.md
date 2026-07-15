# ep2000 drafted decks per gate (s14prod45, checkpoint model_azuki_local_002000.pt)

Two views per gate: the **greedy (argmax) deck** — the exact 50-card list the
checkpoint builds when picking greedily — and the **sampled core** — mean copies
per deck over 24 stochastic drafts (what it treats as core vs flex).
Note: same-element sibling gates build identical greedy decks (the known
draft-time sibling indifference); their differences show up in play, not picks.

## Surge(L) — STT01-002

**Leader:** Raizan &nbsp;•&nbsp; **greedy curve:** 2.22 avg cost &nbsp;•&nbsp; 60% entity / 26% spell / 14% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Recruit | Entity | 1 |
| 1 | Crate Rat Kurobo | Entity | 1 |
| 1 | Elder Hoshin | Entity | 1 |
| 1 | Frida | Entity | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Hidden Dagger | Weapon | 1 |
| 1 | Lightning Orb | Spell | 1 |
| 1 | Lightning Shuriken | Weapon | 1 |
| 1 | Link | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Rei | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | Sundering Strike | Spell | 1 |
| 1 | The Red Bean | Spell | 1 |
| 2 | Alley Fetchduck | Entity | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Alpine Prowler | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Black Jade Pawnbroker | Entity | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Kira | Entity | 1 |
| 2 | Ninpo: Thunderstep | Spell | 1 |
| 2 | Raizan's Riposte | Spell | 1 |
| 2 | Silver Current, Haruhi | Entity | 1 |
| 2 | Stormglass Daggers | Weapon | 1 |
| 3 | Denmu | Entity | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Lightning Kanabo | Weapon | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Raimaru the Stolen | Entity | 1 |
| 3 | Rainy Day Assassin | Entity | 1 |
| 3 | Riven Flashborne | Entity | 1 |
| 3 | Tenraku | Weapon | 1 |
| 4 | Arms Dealer, Kin | Entity | 1 |
| 4 | Caravan Guard | Entity | 1 |
| 4 | Drunken Brewmaster | Entity | 1 |
| 4 | Fermented Beanz | Entity | 1 |
| 4 | Forging Tricks | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 4 | Piko | Entity | 1 |
| 4 | Stormglass Katana | Weapon | 1 |
| 5 | Thunderclap | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.20)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Raizan's Riposte | Spell | 2 |
| 1.00 | Kira | Entity | 2 |
| 1.00 | Lightning Orb | Spell | 1 |
| 1.00 | Lightning Kanabo | Weapon | 3 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | The Red Bean | Spell | 1 |
| 1.00 | Lightning Shuriken | Weapon | 1 |
| 1.00 | Hidden Dagger | Weapon | 1 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Ninpo: Thunderstep | Spell | 2 |
| 1.00 | Sundering Strike | Spell | 1 |
| 1.00 | Stormglass Daggers | Weapon | 2 |
| 1.00 | Mocking Dummy | Entity | 1 |
| 1.00 | Raimaru the Stolen | Entity | 3 |
| 1.00 | Good Enough Replica | Spell | 2 |

Leader split over sampled drafts: STT01-001×13, AZK01-119×11

## Stormchain(L) — AZK01-120

**Leader:** Raizan &nbsp;•&nbsp; **greedy curve:** 2.22 avg cost &nbsp;•&nbsp; 60% entity / 26% spell / 14% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Recruit | Entity | 1 |
| 1 | Crate Rat Kurobo | Entity | 1 |
| 1 | Elder Hoshin | Entity | 1 |
| 1 | Frida | Entity | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Hidden Dagger | Weapon | 1 |
| 1 | Lightning Orb | Spell | 1 |
| 1 | Lightning Shuriken | Weapon | 1 |
| 1 | Link | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Rei | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | Sundering Strike | Spell | 1 |
| 1 | The Red Bean | Spell | 1 |
| 2 | Alley Fetchduck | Entity | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Alpine Prowler | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Black Jade Pawnbroker | Entity | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Kira | Entity | 1 |
| 2 | Ninpo: Thunderstep | Spell | 1 |
| 2 | Raizan's Riposte | Spell | 1 |
| 2 | Silver Current, Haruhi | Entity | 1 |
| 2 | Stormglass Daggers | Weapon | 1 |
| 3 | Denmu | Entity | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Lightning Kanabo | Weapon | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Raimaru the Stolen | Entity | 1 |
| 3 | Rainy Day Assassin | Entity | 1 |
| 3 | Riven Flashborne | Entity | 1 |
| 3 | Tenraku | Weapon | 1 |
| 4 | Arms Dealer, Kin | Entity | 1 |
| 4 | Caravan Guard | Entity | 1 |
| 4 | Drunken Brewmaster | Entity | 1 |
| 4 | Fermented Beanz | Entity | 1 |
| 4 | Forging Tricks | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 4 | Piko | Entity | 1 |
| 4 | Stormglass Katana | Weapon | 1 |
| 5 | Thunderclap | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.20)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Raizan's Riposte | Spell | 2 |
| 1.00 | Kira | Entity | 2 |
| 1.00 | Lightning Orb | Spell | 1 |
| 1.00 | Lightning Kanabo | Weapon | 3 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | The Red Bean | Spell | 1 |
| 1.00 | Lightning Shuriken | Weapon | 1 |
| 1.00 | Hidden Dagger | Weapon | 1 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Ninpo: Thunderstep | Spell | 2 |
| 1.00 | Sundering Strike | Spell | 1 |
| 1.00 | Stormglass Daggers | Weapon | 2 |
| 1.00 | Mocking Dummy | Entity | 1 |
| 1.00 | Raimaru the Stolen | Entity | 3 |
| 1.00 | Good Enough Replica | Spell | 2 |

Leader split over sampled drafts: STT01-001×13, AZK01-119×11

## Hydromancy(W) — STT02-002

**Leader:** Shao &nbsp;•&nbsp; **greedy curve:** 2.16 avg cost &nbsp;•&nbsp; 56% entity / 36% spell / 8% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 0 | Aquatic Veil | Spell | 1 |
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Dagger | Weapon | 1 |
| 1 | Chilling Water | Spell | 1 |
| 1 | Hayabusa Saburo | Entity | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Link | Entity | 1 |
| 1 | Lotus of Paradise | Spell | 1 |
| 1 | Mizuto | Entity | 1 |
| 1 | Mocking Dummy | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Rei | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | The Red Bean | Spell | 1 |
| 1 | Tidal Insight | Spell | 1 |
| 1 | Water Orb | Spell | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Benzai the Merchant | Entity | 1 |
| 2 | Bubble Adept | Entity | 1 |
| 2 | Commune with Water | Spell | 1 |
| 2 | Foamback Crab | Entity | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Lotus of Reflection | Spell | 1 |
| 2 | Mirage Frog | Entity | 1 |
| 2 | Serene Fist, Misaki | Entity | 1 |
| 2 | Tenshin | Weapon | 1 |
| 3 | Bubblemancer | Entity | 1 |
| 3 | Fumiko | Entity | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Maho | Entity | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Rippling Recall | Spell | 1 |
| 3 | Rooftop Hunter | Entity | 1 |
| 3 | Selis of the Shore | Entity | 1 |
| 3 | Stalking Assassin | Entity | 1 |
| 3 | Tenraku | Weapon | 1 |
| 3 | Top Beanz | Entity | 1 |
| 4 | Mizuryuu's Torrent | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 5 | Black Jade Brawler | Entity | 1 |
| 6 | Pulled Under | Spell | 1 |
| 6 | Shao's Perseverance | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.11)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Lotus of Reflection | Spell | 2 |
| 1.00 | Pip | Entity | 1 |
| 1.00 | Lotus of Paradise | Spell | 1 |
| 1.00 | The Red Bean | Spell | 1 |
| 1.00 | Aquatic Veil | Spell | 0 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Chilling Water | Spell | 1 |
| 1.00 | Benzai the Merchant | Entity | 2 |
| 1.00 | Good Enough Replica | Spell | 2 |
| 1.00 | Water Orb | Spell | 1 |
| 1.00 | Commune with Water | Spell | 2 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | Serene Fist, Misaki | Entity | 2 |
| 1.00 | Mocking Dummy | Entity | 1 |
| 1.00 | Tidal Insight | Spell | 1 |

Leader split over sampled drafts: STT02-001×13, AZK01-125×11

## EchoedWaves(W) — AZK01-126

**Leader:** Shao &nbsp;•&nbsp; **greedy curve:** 2.16 avg cost &nbsp;•&nbsp; 56% entity / 36% spell / 8% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 0 | Aquatic Veil | Spell | 1 |
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Dagger | Weapon | 1 |
| 1 | Chilling Water | Spell | 1 |
| 1 | Hayabusa Saburo | Entity | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Link | Entity | 1 |
| 1 | Lotus of Paradise | Spell | 1 |
| 1 | Mizuto | Entity | 1 |
| 1 | Mocking Dummy | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Rei | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | The Red Bean | Spell | 1 |
| 1 | Tidal Insight | Spell | 1 |
| 1 | Water Orb | Spell | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Benzai the Merchant | Entity | 1 |
| 2 | Bubble Adept | Entity | 1 |
| 2 | Commune with Water | Spell | 1 |
| 2 | Foamback Crab | Entity | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Lotus of Reflection | Spell | 1 |
| 2 | Mirage Frog | Entity | 1 |
| 2 | Serene Fist, Misaki | Entity | 1 |
| 2 | Tenshin | Weapon | 1 |
| 3 | Bubblemancer | Entity | 1 |
| 3 | Fumiko | Entity | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Maho | Entity | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Rippling Recall | Spell | 1 |
| 3 | Rooftop Hunter | Entity | 1 |
| 3 | Selis of the Shore | Entity | 1 |
| 3 | Stalking Assassin | Entity | 1 |
| 3 | Tenraku | Weapon | 1 |
| 3 | Top Beanz | Entity | 1 |
| 4 | Mizuryuu's Torrent | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 5 | Black Jade Brawler | Entity | 1 |
| 6 | Pulled Under | Spell | 1 |
| 6 | Shao's Perseverance | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.11)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Lotus of Reflection | Spell | 2 |
| 1.00 | Pip | Entity | 1 |
| 1.00 | Lotus of Paradise | Spell | 1 |
| 1.00 | The Red Bean | Spell | 1 |
| 1.00 | Aquatic Veil | Spell | 0 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Chilling Water | Spell | 1 |
| 1.00 | Benzai the Merchant | Entity | 2 |
| 1.00 | Good Enough Replica | Spell | 2 |
| 1.00 | Water Orb | Spell | 1 |
| 1.00 | Commune with Water | Spell | 2 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | Serene Fist, Misaki | Entity | 2 |
| 1.00 | Mocking Dummy | Entity | 1 |
| 1.00 | Tidal Insight | Spell | 1 |

Leader split over sampled drafts: STT02-001×13, AZK01-125×11

## Rushfire(F) — AZK01-122

**Leader:** Kagoro of the Burnt Path &nbsp;•&nbsp; **greedy curve:** 2.12 avg cost &nbsp;•&nbsp; 66% entity / 28% spell / 6% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Dagger | Weapon | 1 |
| 1 | Cinderwake Seer | Entity | 1 |
| 1 | Detonation Pact | Spell | 1 |
| 1 | Fanatic Kindler | Entity | 1 |
| 1 | Frida | Entity | 1 |
| 1 | Glass Blower, Hokuto | Entity | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Ignition Pact | Spell | 1 |
| 1 | Link | Entity | 1 |
| 1 | Lounge Siren, Saeko | Entity | 1 |
| 1 | Mocking Dummy | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Ruby | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | The Red Bean | Spell | 1 |
| 2 | Alley Fetchduck | Entity | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Black Jade Warlord | Entity | 1 |
| 2 | Cinderwake Pursuer | Entity | 1 |
| 2 | Collateral Burst | Spell | 1 |
| 2 | Crazed Arsonist | Entity | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Spice | Entity | 1 |
| 2 | Tenshin | Weapon | 1 |
| 2 | Wolf Cub | Entity | 1 |
| 3 | Firebrand Renji | Entity | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | JD | Entity | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Lady Emberheart | Entity | 1 |
| 3 | Midnight Courier | Entity | 1 |
| 3 | Omen Peddler | Entity | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Reckless Tinkerer | Entity | 1 |
| 3 | Rooftop Hunter | Entity | 1 |
| 3 | Wrath of Sinder | Spell | 1 |
| 4 | Drunken Brewmaster | Entity | 1 |
| 4 | Fire Orb | Spell | 1 |
| 4 | Forging Tricks | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 4 | Spiteful Raider | Entity | 1 |
| 5 | Firestorm | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.11)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Invigorating Concoction | Spell | 3 |
| 1.00 | Black Jade Warlord | Entity | 2 |
| 1.00 | Ruby | Entity | 1 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | Power of Friendship | Spell | 3 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Black Jade Courier | Entity | 1 |
| 1.00 | Wrath of Sinder | Spell | 3 |
| 1.00 | Crazed Arsonist | Entity | 2 |
| 1.00 | Collateral Burst | Spell | 2 |
| 1.00 | Ignition Pact | Spell | 1 |
| 1.00 | Good Enough Replica | Spell | 2 |
| 1.00 | Lounge Siren, Saeko | Entity | 1 |
| 1.00 | Penny | Entity | 1 |
| 1.00 | The Red Bean | Spell | 1 |

Leader split over sampled drafts: STT04-001×13, AZK01-121×11

## Ragefire(F) — STT04-002

**Leader:** Kagoro of the Burnt Path &nbsp;•&nbsp; **greedy curve:** 2.12 avg cost &nbsp;•&nbsp; 66% entity / 28% spell / 6% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Dagger | Weapon | 1 |
| 1 | Cinderwake Seer | Entity | 1 |
| 1 | Detonation Pact | Spell | 1 |
| 1 | Fanatic Kindler | Entity | 1 |
| 1 | Frida | Entity | 1 |
| 1 | Glass Blower, Hokuto | Entity | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Ignition Pact | Spell | 1 |
| 1 | Link | Entity | 1 |
| 1 | Lounge Siren, Saeko | Entity | 1 |
| 1 | Mocking Dummy | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Ruby | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | The Red Bean | Spell | 1 |
| 2 | Alley Fetchduck | Entity | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Black Jade Warlord | Entity | 1 |
| 2 | Cinderwake Pursuer | Entity | 1 |
| 2 | Collateral Burst | Spell | 1 |
| 2 | Crazed Arsonist | Entity | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Spice | Entity | 1 |
| 2 | Tenshin | Weapon | 1 |
| 2 | Wolf Cub | Entity | 1 |
| 3 | Firebrand Renji | Entity | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | JD | Entity | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Lady Emberheart | Entity | 1 |
| 3 | Midnight Courier | Entity | 1 |
| 3 | Omen Peddler | Entity | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Reckless Tinkerer | Entity | 1 |
| 3 | Rooftop Hunter | Entity | 1 |
| 3 | Wrath of Sinder | Spell | 1 |
| 4 | Drunken Brewmaster | Entity | 1 |
| 4 | Fire Orb | Spell | 1 |
| 4 | Forging Tricks | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 4 | Spiteful Raider | Entity | 1 |
| 5 | Firestorm | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.11)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Invigorating Concoction | Spell | 3 |
| 1.00 | Black Jade Warlord | Entity | 2 |
| 1.00 | Ruby | Entity | 1 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | Power of Friendship | Spell | 3 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Black Jade Courier | Entity | 1 |
| 1.00 | Wrath of Sinder | Spell | 3 |
| 1.00 | Crazed Arsonist | Entity | 2 |
| 1.00 | Collateral Burst | Spell | 2 |
| 1.00 | Ignition Pact | Spell | 1 |
| 1.00 | Good Enough Replica | Spell | 2 |
| 1.00 | Lounge Siren, Saeko | Entity | 1 |
| 1.00 | Penny | Entity | 1 |
| 1.00 | The Red Bean | Spell | 1 |

Leader split over sampled drafts: STT04-001×13, AZK01-121×11

## Devotion(E) — AZK01-124

**Leader:** Goro Graveloth &nbsp;•&nbsp; **greedy curve:** 2.24 avg cost &nbsp;•&nbsp; 60% entity / 32% spell / 8% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 0 | Offering to Stillstone | Spell | 1 |
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Dagger | Weapon | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Koyama Farm Potter | Entity | 1 |
| 1 | Link | Entity | 1 |
| 1 | Mina the Geomancer | Entity | 1 |
| 1 | Mocking Dummy | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Rei | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | Sloth Scarecrow | Entity | 1 |
| 1 | The Red Bean | Spell | 1 |
| 1 | Treetop Scout | Entity | 1 |
| 2 | Alley Fetchduck | Entity | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Cactus Farmer | Entity | 1 |
| 2 | Earth Orb | Spell | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Oathstone | Spell | 1 |
| 2 | Shiko the Priestess | Entity | 1 |
| 2 | Tenshin | Weapon | 1 |
| 2 | Wobbly Cabbage Cart | Entity | 1 |
| 2 | Wolf Cub | Entity | 1 |
| 2 | Wrong Step | Spell | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | JD | Entity | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Koyama Farm Caretaker | Entity | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Rainy Day Assassin | Entity | 1 |
| 3 | Rooftop Hunter | Entity | 1 |
| 3 | Sand Stands Still | Spell | 1 |
| 3 | Tenraku | Weapon | 1 |
| 4 | Caravan Guard | Entity | 1 |
| 4 | Drunken Brewmaster | Entity | 1 |
| 4 | Forging Tricks | Spell | 1 |
| 4 | Jar of Beans | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 4 | Prickly Tumbleweed | Entity | 1 |
| 4 | Shroom Tender | Entity | 1 |
| 5 | Crushing Weight | Spell | 1 |
| 5 | Quicksand | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.17)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Tenshin | Weapon | 2 |
| 1.00 | Invigorating Concoction | Spell | 3 |
| 1.00 | Sand Stands Still | Spell | 3 |
| 1.00 | Beanz Mentor | Entity | 2 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | Power of Friendship | Spell | 3 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Black Jade Courier | Entity | 1 |
| 1.00 | Sloth Scarecrow | Entity | 1 |
| 1.00 | Good Enough Replica | Spell | 2 |
| 1.00 | Mina the Geomancer | Entity | 1 |
| 1.00 | Penny | Entity | 1 |
| 1.00 | Shiko the Priestess | Entity | 2 |
| 1.00 | Offering to Stillstone | Spell | 0 |
| 1.00 | Oathstone | Spell | 2 |

Leader split over sampled drafts: STT03-001×13, AZK01-123×11

## Stonehaven(E) — STT03-002

**Leader:** Goro Graveloth &nbsp;•&nbsp; **greedy curve:** 2.24 avg cost &nbsp;•&nbsp; 60% entity / 32% spell / 8% weapon

### Greedy (argmax) deck — 50 cards

| Cost | Card | Type | Copies |
|-----:|------|------|-------:|
| 0 | Offering to Stillstone | Spell | 1 |
| 1 | Alley Thug | Entity | 1 |
| 1 | Black Jade Courier | Entity | 1 |
| 1 | Black Jade Dagger | Weapon | 1 |
| 1 | Healing Flutter | Spell | 1 |
| 1 | Koyama Farm Potter | Entity | 1 |
| 1 | Link | Entity | 1 |
| 1 | Mina the Geomancer | Entity | 1 |
| 1 | Mocking Dummy | Entity | 1 |
| 1 | Penny | Entity | 1 |
| 1 | Pip | Entity | 1 |
| 1 | Rei | Entity | 1 |
| 1 | Sleight of Hand | Spell | 1 |
| 1 | Sloth Scarecrow | Entity | 1 |
| 1 | The Red Bean | Spell | 1 |
| 1 | Treetop Scout | Entity | 1 |
| 2 | Alley Fetchduck | Entity | 1 |
| 2 | Alley Guy | Entity | 1 |
| 2 | Beanz Mentor | Entity | 1 |
| 2 | Cactus Farmer | Entity | 1 |
| 2 | Earth Orb | Spell | 1 |
| 2 | Good Enough Replica | Spell | 1 |
| 2 | Gurugumi Vanguard | Entity | 1 |
| 2 | Gus | Entity | 1 |
| 2 | Hook Sword Strike | Spell | 1 |
| 2 | Johnny | Entity | 1 |
| 2 | Oathstone | Spell | 1 |
| 2 | Shiko the Priestess | Entity | 1 |
| 2 | Tenshin | Weapon | 1 |
| 2 | Wobbly Cabbage Cart | Entity | 1 |
| 2 | Wolf Cub | Entity | 1 |
| 2 | Wrong Step | Spell | 1 |
| 3 | Invigorating Concoction | Spell | 1 |
| 3 | JD | Entity | 1 |
| 3 | Jay | Entity | 1 |
| 3 | Koyama Farm Caretaker | Entity | 1 |
| 3 | Power of Friendship | Spell | 1 |
| 3 | Rainy Day Assassin | Entity | 1 |
| 3 | Rooftop Hunter | Entity | 1 |
| 3 | Sand Stands Still | Spell | 1 |
| 3 | Tenraku | Weapon | 1 |
| 4 | Caravan Guard | Entity | 1 |
| 4 | Drunken Brewmaster | Entity | 1 |
| 4 | Forging Tricks | Spell | 1 |
| 4 | Jar of Beans | Spell | 1 |
| 4 | Monk Staff of Warding | Weapon | 1 |
| 4 | Prickly Tumbleweed | Entity | 1 |
| 4 | Shroom Tender | Entity | 1 |
| 5 | Crushing Weight | Spell | 1 |
| 5 | Quicksand | Spell | 1 |

### Sampled core — mean copies/deck over 24 drafts (avg cost 2.17)

| Copies/deck | Card | Type | Cost |
|------------:|------|------|-----:|
| 1.00 | Tenshin | Weapon | 2 |
| 1.00 | Invigorating Concoction | Spell | 3 |
| 1.00 | Sand Stands Still | Spell | 3 |
| 1.00 | Beanz Mentor | Entity | 2 |
| 1.00 | Sleight of Hand | Spell | 1 |
| 1.00 | Power of Friendship | Spell | 3 |
| 1.00 | Healing Flutter | Spell | 1 |
| 1.00 | Black Jade Courier | Entity | 1 |
| 1.00 | Sloth Scarecrow | Entity | 1 |
| 1.00 | Good Enough Replica | Spell | 2 |
| 1.00 | Mina the Geomancer | Entity | 1 |
| 1.00 | Penny | Entity | 1 |
| 1.00 | Shiko the Priestess | Entity | 2 |
| 1.00 | Offering to Stillstone | Spell | 0 |
| 1.00 | Oathstone | Spell | 2 |

Leader split over sampled drafts: STT03-001×13, AZK01-123×11
