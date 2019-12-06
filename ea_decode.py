# Importing modules
import re

def decode_options(options):

    dscp = 'Landscapes: '
    for l in options['landscapes']:
        dscp += (l + ' ')
    dscp += ' {}'.format('HUB') if options['HUB'] == 1 else 'No {}'.format('HUB')
    dscp += ' - {}'.format('Capabilities') if options['CAPA'] == 1 else ''
    dscp += ' - {}'.format('OU') if options['OU'] == 1 else ''
    dscp += ' - {}'.format('Platform') if options['PLATF'] == 1 else ''
    return dscp

def options_filename(options):

    filename = '_'.join(landscape for landscape in options['landscapes'])
    filename += '_' + (' {}'.format('HUB') if options['HUB'] == 1 else 'No_{}'.format('HUB'))
    filename += '_' + ('{}'.format('CAPA') if options['CAPA'] == 1 else 'No_{}'.format('CAPA'))
    filename += '_' + ('{}'.format('OU') if options['OU'] == 1 else 'No_{}'.format('OU'))
    filename += '_' + ('{}'.format('PlATF') if options['PLATF'] == 1 else 'No_{}'.format('PLATF'))

    if options['CLASS'] != '':
        filename += '_' + re.sub('[\s|\-|\/]+', '_', options['CLASS'])

    return filename

def options_title(options):

    landscape_text = 'Landscapes ' + ' '.join(landscape for landscape in options['landscapes'])

    return decode_options(options)