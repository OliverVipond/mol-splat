"""
Predefined small biomolecules with their SMILES representations.
"""

biomolecules = {
    # Inorganic / very small
    "water": "O",
    "carbon_dioxide": "O=C=O",
    "oxygen": "O=O",
    "nitric_oxide": "[N]=O",
    "hydrogen_peroxide": "OO",
    "ammonia": "N",
    "hydrogen_sulfide": "S",

    # Simple metabolites
    "glucose": "OC[C@H]1O[C@@H](O)[C@H](O)[C@H](O)[C@H]1O",
    "fructose": "OC[C@H]1O[C@](O)(CO)[C@H](O)[C@H](O)[C@H]1O",
    "ribose": "OC[C@H]1O[C@@H](O)[C@H](O)[C@H]1O",
    "glycerol": "OCC(O)CO",
    "ethanol": "CCO",
    "lactate": "CC(O)C(=O)O",
    "pyruvate": "CC(=O)C(=O)O",
    "acetate": "CC(=O)O",
    "formate": "C(=O)O",

    # Amino acids (free form)
    "glycine": "NCC(=O)O",
    "alanine": "CC(N)C(=O)O",
    "serine": "NC(CO)C(=O)O",
    "cysteine": "NC(CS)C(=O)O",
    "aspartate": "NC(CC(=O)O)C(=O)O",
    "glutamate": "NC(CCC(=O)O)C(=O)O",

    # Nitrogenous bases / nucleosides
    "adenine": "Nc1ncnc2ncnc12",
    "guanine": "Nc1nc2[nH]cnc2c(=O)n1",
    "cytosine": "Nc1ncc(=O)[nH]1",
    "uracil": "O=c1[nH]cc(=O)[nH]1",
    "thymine": "Cc1ncc(=O)[nH]c1=O",
    "adenosine": "Nc1ncnc2n(cnc12)[C@H]1O[C@H](CO)[C@@H](O)[C@H]1O",

    # Energy / phosphate chemistry
    "phosphate": "OP(=O)(O)O",
    "pyrophosphate": "OP(=O)(O)OP(=O)(O)O",
    "atp_fragment": "OP(=O)(O)OP(=O)(O)O",  # deliberately truncated; full ATP >30 heavy atoms

    # Lipid building blocks
    "acetyl_coa_fragment": "CC(=O)S",  # core reactive moiety
    "choline": "C[N+](C)(C)CCO",
    "ethanolamine": "NCCO",

    # Redox / cofactors (small)
    "nicotinamide": "NC(=O)c1ccncc1",
    "ascorbic_acid": "OC[C@H]1OC(=O)C(O)=C(O)[C@H]1O",

    # Neurotransmitters / signaling
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "histamine": "NCCc1ncc[nH]1",
    "acetylcholine": "CC(=O)OCC[N+](C)(C)C",

    # Buffers / common lab-biological molecules
    "urea": "NC(=O)N",
    "imidazole": "c1ncc[nH]1",
    "bicarbonate": "O=C(O)O",
}
