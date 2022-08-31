relation_mapper = {
    'Lead': ['Position',''],
    'Position': ['Lead',''],
    'Claim': ['Position',''],
    'Counterclaim': ['Position',''],
    'Rebuttal': ['Counterclaim',''],
    'Evidence': ['Claim',''],
    'Concluding Statement': ['Claim', 'Evidence']
}