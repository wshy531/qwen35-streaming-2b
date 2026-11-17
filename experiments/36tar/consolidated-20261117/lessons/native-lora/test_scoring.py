"""Content scoring must tolerate wrappers but reject ambiguous letters."""
from analyze import choice
cases=[('B',4,1),('Answer: (B).',4,1),('The answer is B because the cup moves.',4,1),('B or C',4,None),('E',4,None),('',5,None),('Option C',5,2),('I cannot tell.',5,None)]
for text,n,expected in cases:assert choice(text,n)==expected,(text,choice(text,n),expected)
print({'passed':True,'cases':len(cases)})
