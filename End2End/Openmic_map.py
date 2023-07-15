OpenmicIDX2Name ={
    0: 'accordion',
    1: 'banjo',
    2: 'bass',
    3: 'cello',
    4: 'clarinet',
    5: 'cymbals',
    6: 'drums',
    7: 'flute',
    8: 'guitar',
    9: 'mallet_percussion',
    10: 'mandolin',
    11: 'organ',
    12: 'piano',
    13: 'saxophone',
    14: 'synthesizer',
    15: 'trombone',
    16: 'trumpet',
    17: 'ukulele',
    18: 'violin',
    19: 'voice',
    20: 'Empty' # this is for DETR
    }

Name2OpenmicIDX = {}
for idx,name in OpenmicIDX2Name.items():
    Name2OpenmicIDX[name] = idx
    
OpenMic_Class_NUM = len(Name2OpenmicIDX)