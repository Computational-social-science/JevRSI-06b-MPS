import Gate
open Gate
#eval row "no_champion_up" ⟨(55000 : Rat) / 1, none, (55000 : Rat) / 1, .up⟩
#eval row "clears_with_champ" ⟨(345000 : Rat) / 1, some ((200000 : Rat) / 1), (45000 : Rat) / 1, .up⟩
#eval row "fails_with_champ" ⟨(235000 : Rat) / 1, some ((200000 : Rat) / 1), (45000 : Rat) / 1, .up⟩
#eval row "equal_champ_zero_floor" ⟨(245000 : Rat) / 1, some ((200000 : Rat) / 1), (45000 : Rat) / 1, .up⟩
#eval row "down_ignores_champ" ⟨(55000 : Rat) / 1, some ((200000 : Rat) / 1), (45000 : Rat) / 1, .down⟩
#eval row "down_clears" ⟨(-55000 : Rat) / 1, none, (45000 : Rat) / 1, .down⟩
