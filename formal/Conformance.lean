import Gate
open Gate
#eval row "no_champion_up" ⟨(143000 : Rat) / 1, none, (143000 : Rat) / 1, .up⟩
#eval row "clears_with_champ" ⟨(433000 : Rat) / 1, some ((200000 : Rat) / 1), (133000 : Rat) / 1, .up⟩
#eval row "fails_with_champ" ⟨(323000 : Rat) / 1, some ((200000 : Rat) / 1), (133000 : Rat) / 1, .up⟩
#eval row "equal_champ_zero_floor" ⟨(333000 : Rat) / 1, some ((200000 : Rat) / 1), (133000 : Rat) / 1, .up⟩
#eval row "down_ignores_champ" ⟨(143000 : Rat) / 1, some ((200000 : Rat) / 1), (133000 : Rat) / 1, .down⟩
#eval row "down_clears" ⟨(-143000 : Rat) / 1, none, (133000 : Rat) / 1, .down⟩
