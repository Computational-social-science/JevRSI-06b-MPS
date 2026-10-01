import Gate
open Gate
#eval row "no_champion_up" ⟨(107280 : Rat) / 1, none, (107280 : Rat) / 1, .up⟩
#eval row "clears_with_champ" ⟨(397280 : Rat) / 1, some ((200000 : Rat) / 1), (97280 : Rat) / 1, .up⟩
#eval row "fails_with_champ" ⟨(287280 : Rat) / 1, some ((200000 : Rat) / 1), (97280 : Rat) / 1, .up⟩
#eval row "equal_champ_zero_floor" ⟨(297280 : Rat) / 1, some ((200000 : Rat) / 1), (97280 : Rat) / 1, .up⟩
#eval row "down_ignores_champ" ⟨(107280 : Rat) / 1, some ((200000 : Rat) / 1), (97280 : Rat) / 1, .down⟩
#eval row "down_clears" ⟨(-107280 : Rat) / 1, none, (97280 : Rat) / 1, .down⟩
