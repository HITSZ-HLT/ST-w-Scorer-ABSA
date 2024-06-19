def make_quads_seq(example):
    quads_seq = []

    for aspect, opinion, category, sentiment in example['quads']:

        if aspect == 'NULL':
            aspect = 'none'
        if opinion == 'NULL':
            opinion = 'none'

        quad_seq = ' | '.join([category, sentiment, aspect, opinion])
        quads_seq.append(quad_seq)

    return ' ; '.join(quads_seq)


def parse_quad_seq(quad_seq, example):
    if quad_seq.count('|') != 3:
        return False

    category, sentiment, aspect, opinion = quad_seq.split('|')
    
    aspect  = aspect.strip()
    opinion = opinion.strip()
    category = category.strip()
    sentiment = sentiment.strip()

    if aspect == 'none':
        aspect = 'NULL'
    if opinion == 'none':
        opinion = 'NULL'

    if aspect != 'NULL' and (example is not None and aspect not in example['sentence']):
        return False

    if opinion != 'NULL' and (example is not None and opinion not in example['sentence']):
        return False

    if sentiment not in ('positive', 'neutral', 'negative'):
        return False

    if category not in (
        'food style_options', 'drinks quality', 'food quality', 'food prices', 'ambience general', 'restaurant miscellaneous', 'drinks style_options', 'location general', 'restaurant prices', 'service general', 'restaurant general', 'drinks prices',

        'company general', 'os usability', 'battery operation_performance', 'power_supply operation_performance', 'graphics operation_performance', 'laptop price', 'ports general', 'laptop operation_performance', 'hard_disc quality', 'display design_features', 'hardware operation_performance', 'multimedia_devices design_features', 'software quality', 'software design_features', 'fans&cooling general', 'motherboard general', 'hard_disc operation_performance', 'warranty general', 'cpu quality', 'battery general', 'optical_drives general', 'software general', 'hardware quality', 'display operation_performance', 'software usability', 'os quality', 'power_supply design_features', 'ports connectivity', 'laptop portability', 'laptop general', 'power_supply general', 'ports design_features', 'keyboard design_features', 'battery quality', 'laptop design_features', 'multimedia_devices operation_performance', 'laptop quality', 'memory general', 'support quality', 'cpu operation_performance', 'hard_disc design_features', 'power_supply quality', 'hardware usability', 'keyboard operation_performance', 'ports quality', 'keyboard quality', 'fans&cooling quality', 'os general', 'graphics design_features', 'display usability', 'multimedia_devices quality', 'memory usability', 'laptop connectivity', 'cpu general', 'graphics general', 'support general', 'software operation_performance', 'keyboard miscellaneous', 'fans&cooling operation_performance', 'ports usability', 'shipping quality', 'keyboard usability', 'shipping general', 'display quality', 'memory operation_performance', 'support operation_performance', 'memory design_features', 'battery design_features', 'shipping operation_performance', 'mouse usability', 'hardware design_features', 'multimedia_devices general', 'display general', 'keyboard general', 'os operation_performance', 'laptop usability', 'hard_disc general', 'out_of_scope design_features', 'hardware general', 'ports operation_performance', 'os design_features'
    ):

        return False

    return aspect, opinion, category, sentiment



def parse_quads_seq(quads_seq, example=None):
    quads = []
    valid_flag = True

    quad_seqs = quads_seq.split(';')
    for quad_seq in quad_seqs:
        quad = parse_quad_seq(quad_seq.strip(), example)
        if not quad:
            valid_flag = False
        else:
            quads.append(list(quad))

    return quads, valid_flag



def get_quad_aspect_opinion_num(quads_or_quads_seq_or_example):

    if type(quads_or_quads_seq_or_example) is str:
        quads = parse_quads_seq(quads_or_quads_seq_or_example)[0]

    elif type(quads_or_quads_seq_or_example) is dict:
        example = quads_or_quads_seq_or_example
        if 'quads' in example:
            quads = example['quads']
        else:
            quads = parse_quads_seq(example['quads_seq'])[0]
    else:
        quads = quads_or_quads_seq_or_example

    aspects = {quad[0] for quad in quads}
    opinions = {quad[1] for quad in quads}

    q_num = len(quads)
    a_num = len(aspects)
    o_num = len(opinions)

    return q_num, a_num, o_num



