# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2018-09-08
#
# Copyright (C) 2018 Taishi Matsumura
#
import io
import os
import glob
import dash
import json
import time
import base64
import zipfile
import datetime
import PIL.Image
import dash_auth
import dash_table
import numpy as np
import pandas as pd
import scipy.signal
import my_threshold
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

                                        
GROUP_COLORS = ['#ff0000', '#ff7f00', '#e6b422', '#38b48b', '#008000',
                '#89c3eb', '#84a2d4', '#3e62ad', '#0000ff', '#7f00ff',
                '#56256e', '#000000']


DATA_ROOT = '/Volumes/sdb/Research/Drosophila/data/TsukubaRIKEN/'
DATA_ROOT = '/mnt/sdb/Research/Drosophila/data/TsukubaRIKEN/'
DATA_ROOT = '//133.24.88.18/sdb/Research/Drosophila/data/TsukubaUniv/'
DATA_ROOT = '//133.24.88.18/sdb/Research/Drosophila/data/OkayamaUniv/'
DATA_ROOT = '//133.24.88.18/sdb/Research/Drosophila/data/TsukubaRIKEN/'

THETA = 50

THRESH_FUNC = my_threshold.entire_stats


app = dash.Dash('Sapphire')
app.css.append_css(
        {'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})
    

# ================================
#  Definition of the viewer page
# ================================
app.layout = html.Div([
    html.Header([
        html.H1(
            'Sapphire',
            style={'display': 'inline-block', 'margin': '10px'},
        ),
        html.Div(
            os.path.basename(os.path.dirname(DATA_ROOT)),
            style={'display': 'inline-block', 'margin': '10px'},
        ),
    ]),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(id='tab-1', label='Tab 1', value='tab-1', children=[
            html.Div([
                html.Div([
                    'Dataset:',
                    html.Br(),
                    html.Div([
                        dcc.Dropdown(
                            id='env-dropdown',
                            placeholder='Select a dataset...',
                            clearable=False,
                        ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '200px',
                            'vertical-align': 'middle',
                        },
                    ),
                    dcc.RadioItems(
                        id='detect-target',
                        options=[
                            {
                                'label': 'Pupariation',
                                'value': 'pupariation',
                                'disabled': True,
                            },
                            {
                                'label': 'Pupariation&Eclosion',
                                'value': 'pupa-and-eclo',
                                'disabled': True,
                            },
                            {
                                'label': 'Death',
                                'value': 'death',
                                'disabled': True,
                            },
                        ],
                    ),
                    'Inference Data:',
                    html.Br(),
                    html.Div([
                        dcc.Dropdown(
                            id='larva-dropdown',
                            placeholder='Select a larva data...',
                            clearable=False,
                        ),
                        dcc.Dropdown(
                            id='adult-dropdown',
                            placeholder='Select a adult data...',
                            clearable=False,
                        ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '200px',
                            'vertical-align': 'middle',
                        },
                    ),
                    html.Br(),
                    'Well Index:',
                    html.Br(),
                    html.Div([
                        dcc.Input(
                            id='well-selector',
                            type='number',
                            value=0,
                            min=0,
                            size=5,
                        ),
                        ],
                        style={
                            'display': 'inline-block',
                        },
                    ),
                    html.Br(),
                    html.Div([
                        dcc.Slider(
                            id='well-slider',
                            value=0,
                            min=0,
                            step=1,
                        ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '200px',
                        },
                    ),
                    html.Br(),
                    'Frame:',
                    html.Br(),
                    html.Div([
                            dcc.Input(
                                id='time-selector',
                                type='number',
                                value=0,
                                min=0,
                                size=5,
                            ),
                        ],
                        style={
                            'display': 'inline-block',
                        },
                    ),
                    html.Br(),
                    html.Div([
                        dcc.Slider(
                            id='time-slider',
                            value=0,
                            min=0,
                            step=1,
                        ),
                        ],
                        style={
                            'display': 'inline-block',
                            'width': '200px',
                        },
                    ),
                    html.Br(),
                ], style={
                    'display': 'table-cell',
                    'vertical-align': 'top',
                }),

                html.Div([
                    html.Div(id='org-image'),

                    html.Div(id='label-and-prob'),

                    dcc.Checklist(
                        id='blacklist-check',
                        options=[{
                            'label': 'Black List',
                            'value': 'checked',
                            'disabled': False,
                        }],
                        values=[],
                    ),

                    html.Div(id='blacklist-link'),

                ], style={
                    'display': 'table-cell',
                    'vertical-align': 'top',
                }),

                html.Div([
                    html.Div('Original Image at "t"'),

                    dcc.Graph(
                        id='current-well',
                        config={'displayModeBar': False},
                    ),
                ], style={
                    'display': 'table-cell',
                    'vertical-align': 'top',
                }),

                html.Div([
                    'Data root:',
                    html.Div(DATA_ROOT, id='data-root'),
                    ],
                    style={
                        'display': 'none',
                        'vertical-align': 'top',
                        
                    },
                ),

                html.Div([
                    html.Div(id='larva-signal-div', children=[
                        html.Div([
                            html.Div([
                                html.Div([
                                    dcc.Slider(
                                        id='larva-thresh',
                                        value=2,
                                        min=-5,
                                        max=20,
                                        step=.1,
                                        updatemode='mouseup',
                                        vertical=True,
                                    ),
                                ], style={
                                    'height': '170px',
                                    'width': '10px',
                                    'margin': '0px 25px 10px',
                                }),
                                dcc.Input(
                                    id='larva-thresh-selector',
                                    type='number',
                                    value=2,
                                    min=-5,
                                    max=20,
                                    step=0.1,
                                    style={
                                        'width': '70px',
                                    },
                                ),
                            ], style={
                                'display': 'table-cell',
                                'vertical-align': 'top',
                            }),
                            dcc.Graph(
                                id='larva-signal',
                                figure={'data': []},
                                style={
                                    'display': 'table-cell',
                                    'vertical-align': 'top',
                                    'height': '240px',
                                    'width': '550px',
                                },
                            ),
                            html.Div([
                                html.Div('Signal Type:',
                                    style={'margin-left': '10px'},
                                ),
                                html.Div([
                                    dcc.Dropdown(
                                        id='larva-signal-type',
                                        placeholder='Select a signal...',
                                        clearable=False,
                                    ),
                                    ],
                                    style={
                                        'margin-left': '10px',
                                    },
                                ),
                                dcc.Checklist(
                                    id='larva-smoothing-check',
                                    options=[
                                        {'label': 'Smoothing', 'value': True}],
                                    values=[],
                                    style={
                                        'margin-left': '10px',
                                    },
                                ),
                                html.Div([
                                    'Size:',
                                    dcc.Input(
                                        id='larva-window-size',
                                        type='number',
                                        value=10,
                                        min=0,
                                        size=5,
                                        style={
                                            'width': '70px',
                                            'margin-left': '25px',
                                        },
                                    ),
                                ], style={'margin': '0px 0px 0px 20px'}),
                                html.Div([
                                    'Sigma:',
                                    dcc.Input(
                                        id='larva-window-sigma',
                                        type='number',
                                        value=5,
                                        min=0,
                                        size=5,
                                        step=0.1,
                                        style={
                                            'width': '70px',
                                            'margin-left': '10px',
                                        },
                                    ),
                                ], style={'margin': '0px 0px 0px 20px'}),
                                dcc.Checklist(
                                    id='larva-weight-check',
                                    options=[
                                        {'label': 'Weight', 'value': True}],
                                    values=[],
                                    style={
                                        'margin-left': '10px',
                                    },
                                ),
                                dcc.RadioItems(
                                    id='larva-weight-style',
                                    options=[
                                        {
                                            'label': 'Step',
                                            'value': 'step',
                                            'disabled': True,
                                        },
                                        {
                                            'label': 'Ramp',
                                            'value': 'ramp',
                                            'disabled': True,
                                        },
                                    ],
                                    value='step',
                                    labelStyle={'display': 'inline-block'},
                                    style={'margin': '0px 0px 0px 20px'},
                                ),
                            ], style={
                                'display': 'table-cell',
                                'vertical-align': 'middle',
                            }),
                        ], style={
                            'display': 'table',
                            'table-layout': 'auto',
                        }),
                    ], style={'width': '810px', 'margin-top': '10px'}),

                    dcc.Graph(
                        id='larva-thresh-heatmap',
                        figure={'data': [], 'type': 'heatmap'},
                        style={
                            'display': 'table-cell',
                            'vertical-align': 'top',
                            'height': '240px',
                            'width': '550px',
                        },
                    ),

                    html.Div(id='adult-signal-div', children=[
                        html.Div([
                            html.Div([
                                html.Div([
                                    dcc.Slider(
                                        id='adult-thresh',
                                        value=2,
                                        min=-5,
                                        max=20,
                                        step=.1,
                                        updatemode='mouseup',
                                        vertical=True,
                                    ),
                                ], style={
                                    'height': '170px',
                                    'width': '10px',
                                    'margin': '0px 25px 10px',
                                }),
                                dcc.Input(
                                    id='adult-thresh-selector',
                                    type='number',
                                    value=2,
                                    min=-5,
                                    max=20,
                                    step=0.1,
                                    style={
                                        'width': '70px',
                                    },
                                ),
                            ], style={
                                'display': 'table-cell',
                                'vertical-align': 'top',
                            }),
                            dcc.Graph(
                                id='adult-signal',
                                figure={'data': []},
                                style={
                                    'display': 'table-cell',
                                    'vertical-align': 'top',
                                    'height': '240px',
                                    'width': '550px',
                                },
                            ),
                            html.Div([
                                html.Div('Signal Type:',
                                    style={'margin-left': '10px'},
                                ),
                                html.Div([
                                    dcc.Dropdown(
                                        id='adult-signal-type',
                                        placeholder='Select a signal...',
                                        clearable=False,
                                    ),
                                    ],
                                    style={
                                        'margin-left': '10px',
                                    },
                                ),
                                dcc.Checklist(
                                    id='adult-smoothing-check',
                                    options=[
                                        {'label': 'Smoothing', 'value': True}],
                                    values=[],
                                    style={
                                        'margin-left': '10px',
                                    },
                                ),
                                html.Div([
                                    'Size:',
                                    dcc.Input(
                                        id='adult-window-size',
                                        type='number',
                                        value=10,
                                        min=0,
                                        size=5,
                                        style={
                                            'width': '70px',
                                            'margin-left': '25px',
                                        },
                                    ),
                                ], style={'margin': '0px 0px 0px 20px'}),
                                html.Div([
                                    'Sigma:',
                                    dcc.Input(
                                        id='adult-window-sigma',
                                        type='number',
                                        value=5,
                                        min=0,
                                        size=5,
                                        step=0.1,
                                        style={
                                            'width': '70px',
                                            'margin-left': '10px',
                                        },
                                    ),
                                ], style={'margin': '0px 0px 0px 20px'}),
                                dcc.Checklist(
                                    id='adult-weight-check',
                                    options=[
                                        {'label': 'Weight', 'value': True}],
                                    values=[],
                                    style={
                                        'display': 'inline-block',
                                        'margin': '0px 10px',
                                    },
                                ),
                                dcc.RadioItems(
                                    id='adult-weight-style',
                                    options=[
                                        {
                                            'label': 'Step',
                                            'value': 'step',
                                            'disabled': True,
                                        },
                                        {
                                            'label': 'Ramp',
                                            'value': 'ramp',
                                            'disabled': True,
                                        },
                                    ],
                                    value='step',
                                    labelStyle={'display': 'inline-block'},
                                    style={'margin': '0px 0px 0px 20px'},
                                ),
                            ], style={
                                'display': 'table-cell',
                                'vertical-align': 'middle',
                            }),
                        ], style={
                            'display': 'table',
                            'table-layout': 'auto',
                        }),
                    ], style={'width': '810px', 'margin-top': '10px'}),

                    html.Div([
                        dcc.Input(
                            id='midpoint-selector',
                            type='number',
                            min=0,
                            style={
                                'width': '70px',
                                'display': 'inline-block',
                                'vertical-align': 'middle',
                                'margin': '0px 10px 0px 120px',
                            },
                        ),
                        html.Div([
                            dcc.Slider(
                                id='midpoint-slider',
                                min=0,
                                step=1,
                            ),
                        ], style={
                            'width': '450px',
                            'display': 'inline-block',
                        }),
                    ]),
                ], style={
                        'display': 'table-cell',
                        'vertical-align': 'middle',
                }),
            ], style={
                    'display': 'table',
                    'table-layout': 'auto',
                    'border-collapse': 'separate',
                    'border-spacing': '5px 10px',
            }),

            html.Div([
                dcc.Graph(
                    id='larva-summary',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
                dcc.Graph(
                    id='larva-hist',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
                dcc.Graph(
                    id='larva-boxplot',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
            ]),
            html.Br(),
            html.Div([
                dcc.Graph(
                    id='adult-summary',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
                dcc.Graph(
                    id='adult-hist',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
                dcc.Graph(
                    id='adult-boxplot',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
                dcc.Graph(
                    id='pupa-vs-eclo',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
                dcc.Graph(
                    id='survival-curve',
                    style={
                        'display': 'inline-block',
                        'height': '300px',
                        'width': '20%',
                    },
                ),
            ]),
            html.Div(id='dummy-div'),
        ]),
        dcc.Tab(id='tab-2', label='Tab 2', value='tab-2', children=[
            html.Div(
                id='timestamp-table',
                style={
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin': '10px',
                    'width': '200px',
                },
            ),
            html.Div(
                id='larva-man-table',
                style={
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin': '10px',
                    'width': '400px',
                },
            ),
            html.Div(
                id='larva-auto-table',
                style={
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin': '10px',
                    'width': '400px',
                },
            ),
            html.Div(
                id='adult-man-table',
                style={
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin': '10px',
                    'width': '400px',
                },
            ),
            html.Div(
                id='adult-auto-table',
                style={
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin': '10px',
                    'width': '400px',
                },
            ),
        ], style={'width': '100%'}),
    ], style={'width': '100%'}),

    dcc.Store(id='hidden-timestamp'),
    dcc.Store(id='hidden-midpoint'),
    dcc.Store(id='hidden-blacklist'),

    html.Div('{"changed": "nobody"}',
            id='changed-well', style={'display': 'none'}),
    html.Div(id='well-buff', style={'display': 'none'}, children=json.dumps(
            {
                'nobody': 0,
                'current-well': 0,
                'larva-summary': 0,
                'adult-summary': 0,
                'pupa-vs-eclo': 0,
                'larva-boxplot': 0,
                'adult-boxplot': 0,
            }
        )
    ),

    html.Div('{"changed": "nobody"}',
            id='changed-time', style={'display': 'none'}),
    html.Div(id='time-buff', style={'display': 'none'}, children=json.dumps(
            {
                'nobody': 0,
                'larva-signal': 0,
                'adult-signal': 0,
            }
        )
    ),

], style={'width': '1400px',},)


# =========================================
#  Smoothing signals with gaussian window
# =========================================
def my_filter(signals, size=10, sigma=5):
    
    window = scipy.signal.gaussian(size, sigma)

    signals = np.array(
            [np.convolve(signal, window, mode='same')
                for signal in signals])

    return signals


# =================================================
#  Initialize env-dropdown when opening the page.
# =================================================
@app.callback(
        Output('env-dropdown', 'options'),
        [Input('data-root', 'children')])
def callback(data_root):

    imaging_envs = [os.path.basename(i)
            for i in sorted(glob.glob(os.path.join(data_root, '*')))]

    return [{'label': i, 'value': i} for i in imaging_envs]


# =================================================================
#  Initialize detect-target.
# =================================================================
@app.callback(
        Output('detect-target', 'value'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    # Guard
    if env is None:
        return
    if not os.path.exists(os.path.join(data_root, env, 'config.json')):
        return

    with open(os.path.join(data_root, env, 'config.json')) as f:
        config = json.load(f)

    if config['detect'] == 'pupariation':
        return 'pupariation'

    elif config['detect'] == 'pupa&eclo':
        return 'pupa-and-eclo'

    elif config['detect'] == 'death':
        return 'death'

    else:
        return


# ======================================================
#  Initialize larva-dropdown.
# ======================================================
@app.callback(
        Output('larva-dropdown', 'disabled'),
        [Input('detect-target', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(detect, data_root, env):
    if env is None:
        return

    if detect == 'pupariation':
        return False

    elif detect == 'pupa-and-eclo':
        return False

    elif detect == 'death':
        return True


@app.callback(
        Output('larva-dropdown', 'options'),
        [Input('detect-target', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(detect, data_root, env):
    if env is None:
        return []
    if detect == 'death':
        return []

    results = [os.path.basename(i)
            for i in sorted(glob.glob(os.path.join(
                data_root, env, 'inference', 'larva', '*')))
            if os.path.isdir(i)]

    return [{'label': i, 'value': i} for i in results]


@app.callback(
        Output('larva-dropdown', 'value'),
        [Input('larva-dropdown', 'options')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(_, data_root, env):
    return None


# ======================================================
#  Initialize adult-dropdown.
# ======================================================
@app.callback(
        Output('adult-dropdown', 'disabled'),
        [Input('detect-target', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(detect, data_root, env):
    if env is None:
        return

    if detect == 'pupariation':
        return True

    elif detect == 'pupa-and-eclo':
        return False

    elif detect == 'death':
        return False


@app.callback(
        Output('adult-dropdown', 'options'),
        [Input('detect-target', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(detect, data_root, env):
    if env is None:
        return []
    if detect == 'pupariation':
        return []

    results = [os.path.basename(i)
            for i in sorted(glob.glob(os.path.join(
                data_root, env, 'inference', 'adult', '*')))
            if os.path.isdir(i)]

    return [{'label': i, 'value': i} for i in results]


@app.callback(
        Output('adult-dropdown', 'value'),
        [Input('adult-dropdown', 'options')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(_, data_root, env):
    return None


# =====================================================
#  Callbacks for selecting a well
# =====================================================
@app.callback(
        Output('well-selector', 'max'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    if env is None or data_root is None:
        return

    with open(os.path.join(data_root, env, 'mask_params.json')) as f:
        params = json.load(f)

    return params['n-rows'] * params['n-plates'] * params['n-clms'] - 1
        

@app.callback(
        Output('well-selector', 'value'),
        [Input('well-slider', 'value')])
def callback(well_idx):
    return well_idx


@app.callback(
        Output('well-slider', 'max'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    if env is None or data_root is None:
        return

    with open(os.path.join(data_root, env, 'mask_params.json')) as f:
        params = json.load(f)

    return params['n-rows'] * params['n-plates'] * params['n-clms'] - 1


@app.callback(
        Output('well-slider', 'value'),
        [Input('env-dropdown', 'value'),
         Input('well-buff', 'children'),
         Input('larva-signal-type', 'value'),
         Input('adult-signal-type', 'value')],
        [State('changed-well', 'children')])
def callback(_, buff, __, ___, changed_data):
    buff = json.loads(buff)
    changed_data = json.loads(changed_data)['changed']

    return buff[changed_data]


# =====================================================
#  Callbacks to buffer a ID of a clicked well
# =====================================================
@app.callback(
        Output('changed-well', 'children'),
        [Input('current-well', 'clickData'),
         Input('larva-summary', 'clickData'),
         Input('adult-summary', 'clickData'),
         Input('pupa-vs-eclo', 'clickData'),
         Input('larva-boxplot', 'clickData'),
         Input('adult-boxplot', 'clickData')],
        [State('well-buff', 'children')])
def callback(current_well, larva_summary, adult_summary,
        pupa_vs_eclo, larva_boxplot, adult_boxplot, buff):
    # Guard
    if current_well is None and  \
       larva_summary is None and  \
       adult_summary is None and  \
       pupa_vs_eclo is None and  \
       larva_boxplot is None and  \
       adult_boxplot is None:
        return '{"changed": "nobody"}'

    if current_well is None:
        current_well = 0
    else:
        current_well = int(current_well['points'][0]['text'])

    if larva_summary is None:
        larva_summary = 0
    else:
        larva_summary = int(larva_summary['points'][0]['text'])

    if adult_summary is None:
        adult_summary = 0
    else:
        adult_summary = int(adult_summary['points'][0]['text'])

    if pupa_vs_eclo is None:
        pupa_vs_eclo = 0
    else:
        pupa_vs_eclo = int(pupa_vs_eclo['points'][0]['text'])

    if larva_boxplot is None:
        larva_boxplot = 0
    else:
        larva_boxplot = int(larva_boxplot['points'][0]['text'])

    if adult_boxplot is None:
        adult_boxplot = 0
    else:
        adult_boxplot = int(adult_boxplot['points'][0]['text'])

    buff = json.loads(buff)

    if current_well != buff['current-well']:
        return '{"changed": "current-well"}'

    if larva_summary != buff['larva-summary']:
        return '{"changed": "larva-summary"}'

    if adult_summary != buff['adult-summary']:
        return '{"changed": "adult-summary"}'

    if pupa_vs_eclo != buff['pupa-vs-eclo']:
        return '{"changed": "pupa-vs-eclo"}'

    if larva_boxplot != buff['larva-boxplot']:
        return '{"changed": "larva-boxplot"}'

    if adult_boxplot != buff['adult-boxplot']:
        return '{"changed": "adult-boxplot"}'

    return '{"changed": "nobody"}'


@app.callback(
        Output('well-buff', 'children'),
        [Input('changed-well', 'children')],
        [State('current-well', 'clickData'),
         State('larva-summary', 'clickData'),
         State('adult-summary', 'clickData'),
         State('pupa-vs-eclo', 'clickData'),
         State('larva-boxplot', 'clickData'),
         State('adult-boxplot', 'clickData'),
         State('well-buff', 'children')])
def callback(changed_data, current_well, larva_summary, adult_summary,
        pupa_vs_eclo, larva_boxplot, adult_boxplot, buff):

    buff = json.loads(buff)
    changed_data = json.loads(changed_data)['changed']
    print('Previous Value')
    print(buff)

    if changed_data == 'nobody':
        pass

    elif changed_data == 'current-well':
        buff['current-well'] = int(current_well['points'][0]['text'])

    elif changed_data == 'larva-summary':
        buff['larva-summary'] = int(larva_summary['points'][0]['text'])

    elif changed_data == 'adult-summary':
        buff['adult-summary'] = int(adult_summary['points'][0]['text'])

    elif changed_data == 'pupa-vs-eclo':
        buff['pupa-vs-eclo'] = int(pupa_vs_eclo['points'][0]['text'])

    elif changed_data == 'larva-boxplot':
        buff['larva-boxplot'] = int(larva_boxplot['points'][0]['text'])

    elif changed_data == 'adult-boxplot':
        buff['adult-boxplot'] = int(adult_boxplot['points'][0]['text'])

    else:
        # Never evaluated
        pass

    print('Current Value')
    print(buff)
    return json.dumps(buff)


# =====================================================
#  Callbacks for selecting time step (frame)
# =====================================================
@app.callback(
        Output('time-selector', 'max'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    if env is None:
        return

    return len(glob.glob(
            os.path.join(data_root, env, 'original', '*.jpg'))) - 2


@app.callback(
        Output('time-selector', 'value'),
        [Input('time-slider', 'value')])
def callback(timestep):
    return timestep


@app.callback(
        Output('time-slider', 'max'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    if env is None:
        return 100

    return len(glob.glob(
            os.path.join(data_root, env, 'original', '*.jpg'))) - 2


@app.callback(
        Output('time-slider', 'value'),
        [Input('env-dropdown', 'value'),
         Input('time-buff', 'children'),
         Input('larva-signal-type', 'value'),
         Input('adult-signal-type', 'value')],
        [State('changed-time', 'children')])
def callback(_, buff, __, ___, changed_data):
    buff = json.loads(buff)
    changed_data = json.loads(changed_data)['changed']

    return buff[changed_data]


# =====================================================
#  Callbacks to buffer a clicked frame number
# =====================================================
@app.callback(
        Output('changed-time', 'children'),
        [Input('larva-signal', 'clickData'),
         Input('adult-signal', 'clickData')],
        [State('time-buff', 'children')])
def callback(larva_signal, adult_signal, buff):
    # Guard
    if larva_signal is None and adult_signal is None:
        return '{"changed": "nobody"}'

    if larva_signal is None:
        larva_signal = 0
    else:
        larva_signal = larva_signal['points'][0]['x']

    if adult_signal is None:
        adult_signal = 0
    else:
        adult_signal = adult_signal['points'][0]['x']

    buff = json.loads(buff)

    if larva_signal != buff['larva-signal']:
        return '{"changed": "larva-signal"}'

    if adult_signal != buff['adult-signal']:
        return '{"changed": "adult-signal"}'

    return '{"changed": "nobody"}'


@app.callback(
        Output('time-buff', 'children'),
        [Input('changed-time', 'children')],
        [State('larva-signal', 'clickData'),
         State('adult-signal', 'clickData'),
         State('time-buff', 'children')])
def callback(changed_data, larva_signal, adult_signal, buff):

    buff = json.loads(buff)
    changed_data = json.loads(changed_data)['changed']
    print('Previous Value')
    print(buff)

    if changed_data == 'nobody':
        return json.dumps(buff)

    if changed_data == 'larva-signal':
        buff['larva-signal'] = larva_signal['points'][0]['x']
        print('Current Value')
        print(buff)
        return json.dumps(buff)

    if changed_data == 'adult-signal':
        buff['adult-signal'] = adult_signal['points'][0]['x']
        print('Current Value')
        print(buff)
        return json.dumps(buff)


# =====================================================
#  Select signal type
# =====================================================
@app.callback(
        Output('larva-signal-type', 'options'),
        [Input('larva-dropdown', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(larva, data_root, dataset_name):
    # Guard
    if larva is None or dataset_name is None:
        return []

    signal_files = sorted(glob.glob(os.path.join(data_root,
            dataset_name, 'inference', 'larva', larva, '*signals.npy')))
    signal_files = [os.path.basename(file_path) for file_path in signal_files]

    return [{'label': i, 'value': i} for i in signal_files]


@app.callback(
        Output('adult-signal-type', 'options'),
        [Input('adult-dropdown', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(adult, data_root, dataset_name):
    # Guard
    if adult is None or dataset_name is None:
        return []

    signal_files = sorted(glob.glob(os.path.join(data_root,
            dataset_name, 'inference', 'adult', adult, '*signals.npy')))
    signal_files = [os.path.basename(file_path) for file_path in signal_files]

    return [{'label': i, 'value': i} for i in signal_files]


@app.callback(
        Output('larva-signal-type', 'value'),
        [Input('larva-signal-type', 'options')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(options, data_root, dataset_name):
    # Guard
    if options == [] or dataset_name is None:
        return None

    return options[0]['value']


@app.callback(
        Output('adult-signal-type', 'value'),
        [Input('adult-signal-type', 'options')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(options, data_root, dataset_name):
    # Guard
    if options == [] or dataset_name is None:
        return None

    return options[0]['value']


# =====================================================
#  
# =====================================================
@app.callback(
        Output('larva-thresh-heatmap', 'figure'),
        [Input('larva-signal', 'figure')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('well-selector', 'value'),
         State('larva-signal-type', 'value'),
         State('larva-thresh-selector', 'value'),
         State('hidden-midpoint', 'data'),
         State('larva-weight-check', 'values'),
         State('larva-weight-style', 'value'),
         State('larva-smoothing-check', 'values'),
         State('larva-window-size', 'value'),
         State('larva-window-sigma', 'value'),
         State('larva-thresh', 'max'),
         State('larva-thresh', 'min'),
         State('larva-thresh', 'step')])
def callback(_, data_root, dataset_name, detect, larva, well_idx, signal_name, coef, midpoints, weighting, weight_style, smoothing, size, sigma, coef_max, coef_min, coef_step):
    # Guard
    if dataset_name is None or larva is None or signal_name is None:
        print('matsu')
        return {'data': []}

    t1 = time.time()
    # Load the data
    larva_diffs = np.load(os.path.join(data_root,
            dataset_name, 'inference', 'larva', larva, signal_name)).T
    print('1. Elapsed time: {}'.format(time.time() - t1))

    coefs = np.arange(coef_min, coef_max, coef_step)
    thresholds_list = [
            THRESH_FUNC(larva_diffs, coef=coef) for coef in coefs]
    print('2. Elapsed time: {}'.format(time.time() - t1))

    larva_diffs = seasoning(
            larva_diffs, 'larva', detect, size, sigma,
            smooth=len(smoothing) != 0,
            weight=len(weighting) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=weight_style)
    print('3. Elapsed time: {}'.format(time.time() - t1))

    bincounts = []
    for i, thresholds in enumerate(thresholds_list):
        auto_evals = detect_event(larva_diffs, thresholds, 'larva', detect)
        bincount = np.bincount(
                auto_evals[auto_evals > 0], minlength=len(larva_diffs.T))
        bincounts.append(bincount)

    print('4. Elapsed time: {}'.format(time.time() - t1))
    return {
            'data': [
                {
                    'z': list(bincounts),
                    'y': list(coefs),
                    'type': 'surface',
                    'colorscale': 'YlGnBu',
                },
            ],
            'layout': {
                'width': 500,
                'height': 500,
                'margin': go.layout.Margin(l=0, b=0, t=20, r=0),
                'xaxis': {
                    'automargin': True,
                },
                'yaxis': {
                    'automargin': True,
                },
            }
        }


# =====================================================
#  Toggle valid/invalid of window size
# =====================================================
@app.callback(
        Output('larva-window-size', 'disabled'),
        [Input('larva-smoothing-check', 'values')])
def callback(checks):
    if len(checks) == 0:
        return True
    else:
        return False


@app.callback(
        Output('adult-window-size', 'disabled'),
        [Input('adult-smoothing-check', 'values')])
def callback(checks):

    if len(checks) == 0:
        return True

    else:
        return False


# ======================================================
#  Toggle valid/invalid of window sigma
# ======================================================
@app.callback(
        Output('larva-window-sigma', 'disabled'),
        [Input('larva-smoothing-check', 'values')])
def callback(checks):
    if len(checks) == 0:
        return True
    else:
        return False


@app.callback(
        Output('adult-window-sigma', 'disabled'),
        [Input('adult-smoothing-check', 'values')])
def callback(checks):

    if len(checks) == 0:
        return True

    else:
        return False


# =========================================================
#  Toggle valid/invalid of the weight style radio buttons
# =========================================================
@app.callback(
        Output('larva-weight-style', 'options'),
        [Input('larva-weight-check', 'values')])
def callback(checks):
    if len(checks) == 0:
        return [
            {'label': 'Step', 'value': 'step', 'disabled': True},
            {'label': 'Ramp', 'value': 'ramp', 'disabled': True},
        ]
    else:
        return [
            {'label': 'Step', 'value': 'step', 'disabled': False},
            {'label': 'Ramp', 'value': 'ramp', 'disabled': False},
        ]


@app.callback(
        Output('adult-weight-style', 'options'),
        [Input('adult-weight-check', 'values')])
def callback(checks):
    if len(checks) == 0:
        return [
            {'label': 'Step', 'value': 'step', 'disabled': True},
            {'label': 'Ramp', 'value': 'ramp', 'disabled': True},
        ]
    else:
        return [
            {'label': 'Step', 'value': 'step', 'disabled': False},
            {'label': 'Ramp', 'value': 'ramp', 'disabled': False},
        ]


# ======================================
#  Callbacks for blacklist
# ======================================
@app.callback(
        Output('blacklist-check', 'values'),
        [Input('well-selector', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('hidden-blacklist', 'data')])
def callback(well_idx, data_root, env, blacklist):
    if well_idx is None or env is None or blacklist is None:
        return []

    if blacklist['value'][well_idx]:
        return 'checked'

    else:
        return []


@app.callback(
        Output('hidden-blacklist', 'data'),
        [Input('blacklist-check', 'values')],
        [State('hidden-blacklist', 'data'),
         State('well-selector', 'value'),
         State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(check, blacklist, well_idx, data_root, dataset_name):
    # Guard
    if well_idx is None or dataset_name is None:
        return

    # Load a mask params
    with open(os.path.join(data_root, dataset_name, 'mask_params.json')) as f:
        params = json.load(f)
    n_wells = params['n-rows'] * params['n-plates'] * params['n-clms']

    # Initialize the buffer
    if blacklist is None or len(blacklist['value']) != n_wells:
        blacklist, exist = load_blacklist(data_root, dataset_name)
        return {'value': list(blacklist)}

    if check:
        blacklist['value'][well_idx] = True
    else:
        blacklist['value'][well_idx] = False

    return blacklist


@app.callback(
        Output('blacklist-link', 'children'),
        [Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(blacklist, data_root, dataset_name):
    # Guard
    if blacklist is None:
        return 'Now loading...'

    # Load a mask params
    with open(os.path.join(data_root, dataset_name, 'mask_params.json')) as f:
        params = json.load(f)
    n_wells = params['n-rows'] * params['n-plates'] * params['n-clms']

    blacklist_table = np.array(blacklist['value'], dtype=int).reshape(
            params['n-rows'] * params['n-plates'], params['n-clms'])
    df = pd.DataFrame(blacklist_table)

    return [
            html.A(
                'Download the Blacklist',
                download='Blacklist({}).csv'.format(dataset_name[0:20]),
                href='data:text/csv;charset=utf-8,' + df.to_csv(
                        index=False, header=False),
                target='_blank',
            ),
        ]


# ========================
#  Update the org-image.
# ========================
@app.callback(
        Output('org-image', 'children'),
        [Input('time-selector', 'value'),
         Input('well-selector', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(time, well_idx, data_root, env):
    # Exception handling
    if env is None:
        return

    # Load the mask
    mask = np.load(os.path.join(data_root, env, 'mask.npy'))

    # Load an original image
    orgimg_paths = sorted(glob.glob(
            os.path.join(data_root, env, 'original', '*.jpg')))
    orgimg1 = np.array(
            PIL.Image.open(orgimg_paths[time]).convert('L'), dtype=np.uint8)
    orgimg2 = np.array(
            PIL.Image.open(orgimg_paths[time+1]).convert('L'), dtype=np.uint8)

    # Cut out an well image from the original image
    r, c = np.where(mask == well_idx)
    orgimg1 = orgimg1[r.min():r.max(), c.min():c.max()]
    orgimg2 = orgimg2[r.min():r.max(), c.min():c.max()]
    orgimg1 = PIL.Image.fromarray(orgimg1)
    orgimg2 = PIL.Image.fromarray(orgimg2)

    # Buffer the well image as byte stream
    buf1 = io.BytesIO()
    buf2 = io.BytesIO()
    orgimg1.save(buf1, format='JPEG')
    orgimg2.save(buf2, format='JPEG')

    return [
            html.Div('Image at "t"',
                    style={'display': 'inline-block', 'margin-right': '25px'}),
            html.Div('"t+1"', style={'display': 'inline-block'}),
            html.Img(
                src='data:image/jpeg;base64,{}'.format(
                        base64.b64encode(buf1.getvalue()).decode('utf-8')),
                style={
                    'background': '#555555',
                    'height': '65px',
                    'width': '65px',
                    'padding': '5px',
                    'display': 'inline-block',
                },
            ),
            html.Img(
                src='data:image/jpeg;base64,{}'.format(
                        base64.b64encode(buf2.getvalue()).decode('utf-8')),
                style={
                    'background': '#555555',
                    'height': '65px',
                    'width': '65px',
                    'padding': '5px',
                    'display': 'inline-block',
                },
            ),
        ]


# =============================
#  Update the label-and-prob.
# =============================
@app.callback(
        Output('label-and-prob', 'children'),
        [Input('time-selector', 'value'),
         Input('well-selector', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('adult-dropdown', 'value')])
def callback(time, well_idx, data_root, env, detect, larva, adult):
    # Guard
    if env is None:
        return

    if detect == 'pupariation':
        # Guard
        if larva is None:
            return

        # Load a npz file storing prob images
        # and get a prob image
        larva_probs = np.load(os.path.join(
                data_root, env, 'inference', 'larva', larva, 'probs',
                '{:03d}.npz'.format(well_idx)))['arr_0'].astype(np.int32)

        larva_prob_img1 = PIL.Image.fromarray(
                larva_probs[time] / 100 * 255).convert('L')
        larva_prob_img2 = PIL.Image.fromarray(
                larva_probs[time+1] / 100 * 255).convert('L')

        larva_label_img1 = PIL.Image.fromarray(
                ((larva_probs[time] > THETA) * 255).astype(np.uint8)
                ).convert('L')
        larva_label_img2 = PIL.Image.fromarray(
                ((larva_probs[time+1] > THETA) * 255).astype(np.uint8)
                ).convert('L')

        # Buffer the well image as byte stream
        larva_prob_buf1 = io.BytesIO()
        larva_prob_buf2 = io.BytesIO()
        larva_label_buf1 = io.BytesIO()
        larva_label_buf2 = io.BytesIO()
        larva_prob_img1.save(larva_prob_buf1, format='JPEG')
        larva_prob_img2.save(larva_prob_buf2, format='JPEG')
        larva_label_img1.save(larva_label_buf1, format='JPEG')
        larva_label_img2.save(larva_label_buf2, format='JPEG')

        data = [
            html.Div('Larva'),
            html.Div([
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                larva_label_buf1.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                larva_label_buf2.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
            ]),
            html.Div([
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                larva_prob_buf1.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                larva_prob_buf2.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
            ]),
        ]

    elif detect == 'pupa-and-eclo':
        data = []

        if larva is None:
            pass

        else:
            # Load a npz file storing prob images
            # and get a prob image
            larva_probs = np.load(os.path.join(
                    data_root, env, 'inference', 'larva', larva, 'probs',
                    '{:03d}.npz'.format(well_idx)))['arr_0'].astype(np.int32)

            larva_prob_img1 = PIL.Image.fromarray(
                    larva_probs[time] / 100 * 255).convert('L')
            larva_prob_img2 = PIL.Image.fromarray(
                    larva_probs[time+1] / 100 * 255).convert('L')

            larva_label_img1 = PIL.Image.fromarray(
                    ((larva_probs[time] > THETA) * 255).astype(np.uint8)
                    ).convert('L')
            larva_label_img2 = PIL.Image.fromarray(
                    ((larva_probs[time+1] > THETA) * 255).astype(np.uint8)
                    ).convert('L')

            # Buffer the well image as byte stream
            larva_prob_buf1 = io.BytesIO()
            larva_prob_buf2 = io.BytesIO()
            larva_label_buf1 = io.BytesIO()
            larva_label_buf2 = io.BytesIO()
            larva_prob_img1.save(larva_prob_buf1, format='JPEG')
            larva_prob_img2.save(larva_prob_buf2, format='JPEG')
            larva_label_img1.save(larva_label_buf1, format='JPEG')
            larva_label_img2.save(larva_label_buf2, format='JPEG')

            data = data + [
                html.Div('Larva'),
                html.Div([
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    larva_label_buf1.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    larva_label_buf2.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                ]),
                html.Div([
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    larva_prob_buf1.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    larva_prob_buf2.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                ]),
            ]

        if adult is None:
            pass

        else:
            adult_probs = np.load(os.path.join(
                    data_root, env, 'inference', 'adult', adult, 'probs',
                    '{:03d}.npz'.format(well_idx)))['arr_0'].astype(np.int32)

            adult_prob_img1 = PIL.Image.fromarray(
                    adult_probs[time] / 100 * 255).convert('L')
            adult_prob_img2 = PIL.Image.fromarray(
                    adult_probs[time+1] / 100 * 255).convert('L')

            adult_label_img1 = PIL.Image.fromarray(
                    ((adult_probs[time] > THETA) * 255).astype(np.uint8)
                    ).convert('L')
            adult_label_img2 = PIL.Image.fromarray(
                    ((adult_probs[time+1] > THETA) * 255).astype(np.uint8)
                    ).convert('L')

            adult_prob_buf1 = io.BytesIO()
            adult_prob_buf2 = io.BytesIO()
            adult_label_buf1 = io.BytesIO()
            adult_label_buf2 = io.BytesIO()
            adult_prob_img1.save(adult_prob_buf1, format='JPEG')
            adult_prob_img2.save(adult_prob_buf2, format='JPEG')
            adult_label_img1.save(adult_label_buf1, format='JPEG')
            adult_label_img2.save(adult_label_buf2, format='JPEG')

            data = data + [
                html.Div('Adult'),
                html.Div([
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    adult_label_buf1.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    adult_label_buf2.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                ]),
                html.Div([
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    adult_prob_buf1.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                    html.Img(
                        src='data:image/jpeg;base64,{}'.format(
                                base64.b64encode(
                                    adult_prob_buf2.getvalue()).decode('utf-8')),
                        style={
                            'background': '#555555',
                            'height': '65px',
                            'width': '65px',
                            'padding': '5px',
                            'display': 'inline-block',
                        },
                    ),
                ]),
            ]

    elif detect == 'death':
        # Guard
        if adult is None:
            return

        # Load a npz file storing prob images
        # and get a prob image
        adult_probs = np.load(os.path.join(
                data_root, env, 'inference', 'adult', adult, 'probs',
                '{:03d}.npz'.format(well_idx)))['arr_0'].astype(np.int32)

        adult_prob_img1 = PIL.Image.fromarray(
                adult_probs[time] / 100 * 255).convert('L')
        adult_prob_img2 = PIL.Image.fromarray(
                adult_probs[time+1] / 100 * 255).convert('L')

        adult_label_img1 = PIL.Image.fromarray(
                ((adult_probs[time] > THETA) * 255).astype(np.uint8)
                ).convert('L')
        adult_label_img2 = PIL.Image.fromarray(
                ((adult_probs[time+1] > THETA) * 255).astype(np.uint8)
                ).convert('L')

        # Buffer the well image as byte stream
        adult_prob_buf1 = io.BytesIO()
        adult_prob_buf2 = io.BytesIO()
        adult_label_buf1 = io.BytesIO()
        adult_label_buf2 = io.BytesIO()
        adult_prob_img1.save(adult_prob_buf1, format='JPEG')
        adult_prob_img2.save(adult_prob_buf2, format='JPEG')
        adult_label_img1.save(adult_label_buf1, format='JPEG')
        adult_label_img2.save(adult_label_buf2, format='JPEG')

        data = [
            html.Div('Adult'),
            html.Div([
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                adult_label_buf1.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                adult_label_buf2.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
            ]),
            html.Div([
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                adult_prob_buf1.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(
                            base64.b64encode(
                                adult_prob_buf2.getvalue()).decode('utf-8')),
                    style={
                        'background': '#555555',
                        'height': '65px',
                        'width': '65px',
                        'padding': '5px',
                        'display': 'inline-block',
                    },
                ),
            ]),
        ]

    return data


# ===========================
#  Update the current-well.
# ===========================
@app.callback(
        Output('current-well', 'figure'),
        [Input('time-selector', 'value'),
         Input('well-selector', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(time, well_idx, data_root, env):
    # Guard
    if env is None:
        return {'data': []}

    # Load a mask params
    with open(os.path.join(data_root, env, 'mask_params.json')) as f:
        params = json.load(f)

    n_wells = params['n-rows'] * params['n-plates'] * params['n-clms']

    xs, ys = well_coordinates(params)

    # Load an original image
    orgimg_paths = sorted(glob.glob(
            os.path.join(data_root, env, 'original', '*.jpg')))
    org_img = PIL.Image.open(orgimg_paths[time]).convert('L')

    # Buffer the well image as byte stream
    buf = io.BytesIO()
    org_img.save(buf, format='JPEG')
    data_uri = 'data:image/jpeg;base64,{}'.format(
            base64.b64encode(buf.getvalue()).decode('utf-8'))

    height, width = np.array(org_img).shape

    # A coordinate of selected well
    selected_x = xs[well_idx:well_idx+1]
    selected_y = ys[well_idx:well_idx+1]

    # Bounding boxes of groups
    if os.path.exists(os.path.join(data_root, env, 'grouping.csv')):
        mask = np.load(os.path.join(data_root, env, 'mask.npy'))
        mask = np.flipud(mask)

        groups = np.loadtxt(
                os.path.join(data_root, env, 'grouping.csv'),
                dtype=np.int16, delimiter=',').flatten()

        for well_idx, group_id in enumerate(groups):
            mask[mask==well_idx] = group_id

        bounding_boxes = [
                {
                    'x': [
                        np.where(mask == group_id)[1].min(),
                        np.where(mask == group_id)[1].max(),
                        np.where(mask == group_id)[1].max(),
                        np.where(mask == group_id)[1].min(),
                        np.where(mask == group_id)[1].min(),
                    ],
                    'y': [
                        np.where(mask == group_id)[0].min(),
                        np.where(mask == group_id)[0].min(),
                        np.where(mask == group_id)[0].max(),
                        np.where(mask == group_id)[0].max(),
                        np.where(mask == group_id)[0].min(),
                    ],
                    'name': 'Group{}'.format(group_id),
                    'mode': 'lines',
                    'line': {'width': 3, 'color': GROUP_COLORS[group_id - 1]},
                }
                for group_id in np.unique(groups)
            ]

        well_points = [
                {
                    'x': xs[groups == group_id],
                    'y': ys[groups == group_id],
                    'text': np.where(groups == group_id)[0].astype(str),
                    'mode': 'markers',
                    'marker': {
                        'size': 4,
                        'color': GROUP_COLORS[group_id - 1],
                        'opacity': 0.0,
                    },
                    'name': 'Group{}'.format(group_id),
                }
                for group_id in np.unique(groups)
            ]

    else:
        bounding_boxes = []

        well_points = [
                {
                    'x': xs,
                    'y': ys,
                    'text': [str(i) for i in range(n_wells)],
                    'mode': 'markers',
                    'marker': {
                        'size': 4,
                        'color': '#ffffff',
                        'opacity': 0.0,
                    },
                    'name': '',
                },
            ]

    return {
            'data': bounding_boxes + well_points + [
                {
                    'x': selected_x,
                    'y': selected_y,
                    'text': str(well_idx),
                    'mode': 'markers',
                    'marker': {'size': 10, 'color': '#ff0000', 'opacity': 0.5},
                    'name': 'Selected well',
                },
            ],
            'layout': {
                'width': 200,
                'height': 400,
                'margin': go.layout.Margin(l=0, b=0, t=0, r=0),
                'xaxis': {
                    'range': (0, width),
                    'scaleanchor': 'y',
                    'scaleratio': 1,
                    'showgrid': False,
                },
                'yaxis': {
                    'range': (0, height),
                    'showgrid': False,
                },
                'images': [{
                    'xref': 'x',
                    'yref': 'y',
                    'x': 0,
                    'y': 0,
                    'yanchor': 'bottom',
                    'sizing': 'stretch',
                    'sizex': width,
                    'sizey': height,
                    'layer': 'below',
                    'source': data_uri,
                }],
                'dragmode': 'zoom',
                'hovermode': 'closest',
                'showlegend': False,
            }
        }


def well_coordinates(params):

    n_rows = params['n-rows']
    n_clms = params['n-clms']
    n_plates = params['n-plates']
    row_gap = params['row-gap']
    clm_gap = params['clm-gap']
    plate_gap = params['plate-gap']
    x = params['x']
    y = params['y']
    well_w = params['well-w']
    well_h = params['well-h']
    angle = np.deg2rad(params['angle'])

    well_idxs = np.flipud(
            np.arange(n_rows * n_clms * n_plates, dtype=int).reshape(
                n_rows*n_plates, n_clms)).reshape(n_rows * n_clms * n_plates)

    xs = []
    ys = []
    count = 0
    for n in range(n_plates):
        for idx_r in range(n_rows):
            for idx_c in range(n_clms):
                c1 = x + round(idx_c*(well_w + clm_gap))
                r1 = y + round(idx_r*(well_h + row_gap))  \
                       + n*(n_rows*well_h + plate_gap)  \
                       + round(row_gap*(n - 1))
                c1, r1 = np.dot(
                        np.array(
                            [[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle),  np.cos(angle)]]),
                        np.array([c1-x, r1-y])) + np.array([x, y])
                c1, r1 = np.round([c1, r1]).astype(int)
                xs.append(c1 + well_w / 2)
                ys.append(r1 + well_h / 2)
                count += 1

    return np.array(xs)[well_idxs], np.array(ys)[well_idxs]


# ======================================================
#  Callbacks for threshold
# ======================================================
@app.callback(
        Output('larva-thresh-selector', 'value'),
        [Input('larva-thresh', 'value')])
def callback(threshold):
    return threshold


@app.callback(
        Output('adult-thresh-selector', 'value'),
        [Input('adult-thresh', 'value')])
def callback(threshold):
    return threshold


# ======================================================
#  Callbacks for midpoint
# ======================================================
@app.callback(
        Output('midpoint-slider', 'max'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    if env is None:
        return 100

    return len(glob.glob(
            os.path.join(data_root, env, 'original', '*.jpg'))) - 2


@app.callback(
        Output('midpoint-slider', 'value'),
        [Input('well-selector', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('hidden-midpoint', 'data')])
def callback(well_idx, data_root, dataset_name, midpoints):
    if well_idx is None or dataset_name is None or midpoints is None:
        return 50

    return midpoints['midpoint'][well_idx]


@app.callback(
        Output('midpoint-selector', 'max'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    if env is None:
        return 100

    return len(glob.glob(
            os.path.join(data_root, env, 'original', '*.jpg'))) - 2


@app.callback(
        Output('midpoint-selector', 'value'),
        [Input('midpoint-slider', 'value')])
def callback(midpoint):
    return midpoint


@app.callback(
        Output('hidden-midpoint', 'data'),
        [Input('midpoint-selector', 'value')],
        [State('hidden-midpoint', 'data'),
         State('well-selector', 'value'),
         State('data-root', 'children'),
         State('env-dropdown', 'value')])
def callback(midpoint, midpoints, well_idx, data_root, dataset_name):
    # Guard
    if well_idx is None or dataset_name is None:
        return

    # Load a mask params
    with open(os.path.join(data_root, dataset_name, 'mask_params.json')) as f:
        params = json.load(f)
    n_wells = params['n-rows'] * params['n-plates'] * params['n-clms']

    # Initialize the buffer
    if midpoints is None or len(midpoints['midpoint']) != n_wells:
        midpoints = {'midpoint': [50] * n_wells}
        return midpoints

    if os.path.exists(os.path.join(data_root, dataset_name, 'grouping.csv')):
        group_ids = np.loadtxt(
                os.path.join(data_root, dataset_name, 'grouping.csv'),
                dtype=np.int16, delimiter=',').flatten()

        midpoints['midpoint'] = np.array(midpoints['midpoint'])
        group_id = group_ids[well_idx]
        midpoints['midpoint'][group_ids == group_id] = midpoint

        return midpoints

    else:
        return {'midpoint': [midpoint] * n_wells}


# =========================================
#  Update the figure in the larva-signal
# =========================================
@app.callback(
        Output('larva-signal', 'figure'),
        [Input('well-selector', 'value'),
         Input('larva-thresh-selector', 'value'),
         Input('time-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value')],
        [State('larva-signal', 'figure'),
         State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('hidden-timestamp', 'data'),
         State('larva-signal-type', 'value')])
def callback(well_idx, coef, time, midpoints, weight, style, checks, size,
        sigma, figure, data_root, env, detect, larva, timestamps, signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if larva is None:
        return {'data': []}
    if len(figure['data']) == 0:
        x, y = 0, 0
    else:
        x, y = time, figure['data'][0]['y'][time]

    larva_data = []
    manual_data = []
    common_data = []

    # Load the data
    larva_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)).T

    diffs_mean = larva_diffs.mean()
    diffs_std = larva_diffs.std()
    thresholds = THRESH_FUNC(larva_diffs, coef=coef)

    larva_diffs = seasoning(
            larva_diffs, 'larva', detect, size, sigma,
            smooth=len(checks) != 0,
            weight=len(weight) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=style)

    auto_evals = detect_event(larva_diffs, thresholds, 'larva', detect)

    if os.path.exists(
            os.path.join(data_root, env, 'original', 'pupariation.csv')):

        manual_evals = np.loadtxt(
                os.path.join(
                    data_root, env, 'original', 'pupariation.csv'),
                dtype=np.int16, delimiter=',').flatten()

        manual_data = [
            {
                # Manual evaluation time (vertical line)
                'x': [manual_evals[well_idx], manual_evals[well_idx]],
                'y': [0, larva_diffs.max()],
                'mode': 'lines',
                'name': 'Manual',
                'line': {'width': 2, 'color': '#ffa500'},
            },
        ]

    larva_data = [
            {
                # Signal
                'x': list(range(len(larva_diffs[0, :]))),
                'y': list(larva_diffs[well_idx]),
                'mode': 'lines',
                'marker': {'color': '#4169e1'},
                'name': 'Signal',
                'opacity':1.0,
                'line': {'width': 2, 'color': '#4169e1'},
            },
            {
                # Threshold (horizontal line)
                'x': [0, len(larva_diffs[0, :])],
                'y': [thresholds[well_idx, 0], thresholds[well_idx, 0]],
                'mode': 'lines',
                'name': 'Threshold',
                'line': {'width': 2, 'color': '#4169e1'},
            },
            {
                # Auto evaluation time (vertical line)
                'x': [auto_evals[well_idx], auto_evals[well_idx]],
                'y': [0, larva_diffs.max()],            
                'name': 'Auto',
                'mode':'lines',
                'line': {'width': 2, 'color': '#4169e1', 'dash': 'dot'},
            },
        ]

    common_data = [
            {
                # Selected data point
                'x': [x],
                'y': [y],
                'mode': 'markers',
                'marker': {'size': 10, 'color': '#ff0000'},
                'name': '',
            },
        ]

    return {
            'data': larva_data + manual_data + common_data,
            'layout': {
                    'annotations': [
                        {
                            'x': 0.01 * len(larva_diffs.T),
                            'y': 1.0 * larva_diffs.max(),
                            'text':
                                'Threshold: {:.1f}'.format(
                                        thresholds[well_idx, 0]) +  \
                                 '={:.1f}'.format(diffs_mean) +  \
                                 '{:+.1f}'.format(coef) +  \
                                 '*{:.1f}'.format(diffs_std),
                            'showarrow': False,
                            'xanchor': 'left',
                        },
                    ],
                    'font': {'size': 15},
                    'xaxis': {
                        'title': 'Frame',
                        'tickfont': {'size': 15},
                    },
                    'yaxis': {
                        'title': 'Diff. of Larva ROI',
                        'tickfont': {'size': 15},
                        'overlaying': 'y',
                        'range': [-0.1*larva_diffs.max(), larva_diffs.max()],

                    },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=70, r=0, b=50, t=10, pad=0),
                'shapes': day_and_night(timestamps),
            },
        }


@app.callback(
        Output('larva-signal-div', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {
                'width': '810px',
                'margin-top': '10px',
            }

    elif detect == 'pupa-and-eclo':
        return {
                'width': '810px',
                'margin-top': '10px',
            }

    elif detect == 'death':
        return {'display': 'none'}

    else:
        return {}


# =========================================
#  Update the figure in the adult-signal.
# =========================================
@app.callback(
        Output('adult-signal', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('adult-thresh-selector', 'value'),
         Input('time-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('adult-weight-check', 'values'),
         Input('adult-weight-style', 'value'),
         Input('adult-smoothing-check', 'values'),
         Input('adult-window-size', 'value'),
         Input('adult-window-sigma', 'value')],
        [State('well-selector', 'value'),
         State('adult-signal', 'figure'),
         State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('adult-dropdown', 'value'),
         State('hidden-timestamp', 'data'),
         State('larva-signal-type', 'value'),
         State('adult-signal-type', 'value')])
def callback(larva_coef, adult_coef, time, midpoints,
        larva_weighting, larva_w_style, larva_smoothing, larva_w_size,
        larva_w_sigma, adult_weighting, adult_w_style, adult_smoothing,
        adult_w_size, adult_w_sigma, well_idx, figure, data_root, env, detect,
        larva, adult, timestamps, larva_signal_name, adult_signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if adult is None:
        return {'data': []}
    if len(figure['data']) == 0:
        x, y = 0, 0
    else:
        x, y = time, figure['data'][0]['y'][time]

    adult_data = []
    manual_data = []
    common_data = []


    # ----------------------------------------------------------
    #  Detect pupariation timing for detecting eclosion timing
    # ----------------------------------------------------------
    if larva is None:
        pupar_times = None

    else:
        # Load the data
        larva_diffs = np.load(os.path.join(data_root,
                env, 'inference', 'larva', larva, larva_signal_name)).T

        larva_thresh = THRESH_FUNC(larva_diffs, coef=larva_coef)

        larva_diffs = seasoning(
                larva_diffs, 'larva', detect, larva_w_size, larva_w_sigma,
                smooth=len(larva_smoothing) != 0,
                weight=len(larva_weighting) != 0,
                pupar_times=None,
                midpoints=midpoints,
                weight_style=larva_w_style)

        pupar_times = detect_event(larva_diffs, larva_thresh, 'larva', detect)


    # ----------------------------------------
    #  Detection of eclosion or death timing
    # ----------------------------------------
    # Load the data
    adult_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)).T
    diffs_mean = adult_diffs.mean()
    diffs_std = adult_diffs.std()
    adult_thresh = THRESH_FUNC(adult_diffs, coef=adult_coef)

    adult_diffs = seasoning(
            adult_diffs, 'adult', detect, adult_w_size, adult_w_sigma,
            smooth=len(adult_smoothing) != 0,
            weight=len(adult_weighting) != 0,
            pupar_times=pupar_times,
            midpoints=midpoints,
            weight_style=adult_w_style)

    auto_evals = detect_event(adult_diffs, adult_thresh, 'adult', detect)

    # Load a manual evaluation of event timing
    if detect == 'pupa-and-eclo' and os.path.exists(
            os.path.join(data_root, env, 'original', 'eclosion.csv')):

        manual_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'eclosion.csv'),
                dtype=np.int16, delimiter=',').flatten()

        manual_data = [
            {
                # Manual evaluation time (vertical line)
                'x': [manual_evals[well_idx], manual_evals[well_idx]],
                'y': [0, adult_diffs.max()],
                'mode': 'lines',
                'name': 'Manual',
                'line': {'width': 2, 'color': '#ffa500'},
            },
        ]

    elif detect == 'death' and os.path.exists(
            os.path.join(data_root, env, 'original', 'death.csv')):

        manual_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'death.csv'),
                dtype=np.int16, delimiter=',').flatten()

        manual_data = [
            {
                # Manual evaluation time (vertical line)
                'x': [manual_evals[well_idx], manual_evals[well_idx]],
                'y': [0, adult_diffs.max()],
                'mode': 'lines',
                'name': 'Manual',
                'line': {'width': 2, 'color': '#ffa500'},
            },
        ]

    adult_data = [
            {
                # Signal
                'x': list(range(len(adult_diffs[0, :]))),
                'y': list(adult_diffs[well_idx]),
                'mode': 'lines',
                'marker': {'color': '#4169e1'},
                'name': 'Signal',
                'opacity':1.0,
                'line': {'width': 2, 'color': '#4169e1'},
            },
            {
                # Threshold (horizontal line)
                'x': [0, len(adult_diffs[0, :])],
                'y': [adult_thresh[well_idx, 0], adult_thresh[well_idx, 0]],
                'mode': 'lines',
                'name': 'Threshold',
                'line': {'width': 2, 'color': '#4169e1'},
            },
            {
                # Auto evaluation time (vertical line)
                'x': [auto_evals[well_idx], auto_evals[well_idx]],
                'y': [0, adult_diffs.max()],            
                'name': 'Auto',
                'mode':'lines',
                'line': {'width': 2, 'color': '#4169e1', 'dash': 'dot'},
            },
        ]

    common_data = [
            {
                # Selected data point
                'x': [x],
                'y': [y],
                'mode': 'markers',
                'marker': {'size': 10, 'color': '#ff0000'},
                'name': '',
            },
        ]

    return {
            'data': adult_data + manual_data + common_data,
            'layout': {
                'annotations': [
                    {
                        'x': 0.01 * len(adult_diffs.T),
                        'y': 1.0 * adult_diffs.max(),
                        'text':
                            'Threshold: {:.1f}'.format(
                                    adult_thresh[well_idx, 0]) +  \
                             '={:.1f}'.format(diffs_mean) +  \
                             '{:+.1f}'.format(adult_coef) +  \
                             '*{:.1f}'.format(diffs_std),
                        'showarrow': False,
                        'xanchor': 'left',
                    },
                ],
                'font': {'size': 15},
                'xaxis': {
                    'title': 'Frame',
                    'tickfont': {'size': 15},
                },
                'yaxis': {
                    'title':'Diff. of Adult ROI',
                    'tickfont': {'size': 15},
                    'side': 'left',
                    'range': [-0.1*adult_diffs.max(), adult_diffs.max()],
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=70, r=0, b=50, t=10, pad=0),
                'shapes': day_and_night(timestamps),
            },
        }


@app.callback(
        Output('adult-signal-div', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}
        
    elif detect == 'pupa-and-eclo':
        return {
                'width': '810px',
                'margin-top': '10px',
            }
        
    elif detect == 'death':
        return {
                'width': '810px',
                'margin-top': '10px',
            }

    else:
        return {}


# =========================================
#  Update the figure in the larva-summary
# =========================================
@app.callback(
        Output('larva-summary', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('larva-signal-type', 'value')])
def callback(coef, well_idx, midpoints, weight, style, checks, size, sigma,
        blacklist, data_root, env, detect, larva, signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if larva is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)):
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'original', 'pupariation.csv')):
        return {'data': []}

    # Load a manual evaluation of event timing
    if not os.path.exists(os.path.join(
                data_root, env, 'original', 'pupariation.csv')):
            #return {'data': []}
            non_manualdata = {'layout': {
                        'annotations': [
                        {
                        'x': 5.0,
                        'y': 2.0,
                        'text': 'Not Available',
                        'showarrow': False,
                        'xanchor': 'right',
                    },]}}
            return non_manualdata

    manual_evals = np.loadtxt(
            os.path.join(data_root, env, 'original', 'pupariation.csv'),
            dtype=np.int16, delimiter=',').flatten()

    # Target wells will be evaluated
    exceptions = np.logical_or(blacklist['value'], manual_evals == 0)
    targets = np.logical_not(exceptions)

    # Load a group table
    group_tables = load_grouping_csv(data_root, env)

    # Load a manual data
    manual_evals = np.loadtxt(
            os.path.join(data_root, env, 'original', 'pupariation.csv'),
            dtype=np.int16, delimiter=',').flatten()

    # Load the data
    larva_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)).T

    thresholds = THRESH_FUNC(larva_diffs, coef=coef)

    larva_diffs = seasoning(
            larva_diffs, 'larva', detect, size, sigma,
            smooth=len(checks) != 0,
            weight=len(weight) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=style)

    auto_evals = detect_event(larva_diffs, thresholds, 'larva', detect)

    # Calculate how many frames auto-evaluation is far from manual's one
    errors = auto_evals[targets] - manual_evals[targets]

    # Calculate the root mean square
    rms = np.sqrt((errors**2).sum() / len(errors))

    # Create data points
    if group_tables == []:

        data_list = [
                {
                    'x': list(auto_evals[exceptions]),
                    'y': list(manual_evals[exceptions]),
                    'text': [str(i) for i in np.where(exceptions)[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#000000'},
                    'name': 'Exceptions',
                },
                {
                    'x': list(auto_evals[targets]),
                    'y': list(manual_evals[targets]),
                    'text': [str(i) for i in np.where(targets)[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#1f77b4'},
                },
            ]

    else:

        data_list = []

        for group_idx, group_table in enumerate(group_tables):

            data_list.append(
                {
                    'x': list(
                        auto_evals[np.logical_and(targets, group_table)]),
                    'y': list(
                        manual_evals[np.logical_and(targets, group_table)]),
                    'text': [str(i)
                        for i in np.where(
                            np.logical_and(targets, group_table))[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': GROUP_COLORS[group_idx]},
                    'name': 'Group{}'.format(group_idx + 1),
                })

            data_list.append(
                {
                    'x': list(
                        auto_evals[np.logical_and(exceptions, group_table)]),
                    'y': list(
                        manual_evals[np.logical_and(exceptions, group_table)]),
                    'text': [str(i)
                        for i in np.where(
                            np.logical_and(exceptions, group_table))[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#000000'},
                    'name': 'Group{}<br>Exceptions'.format(group_idx + 1),
                })



    return {
            'data': [
                {
                    'x': [
                        round(0.05 * len(larva_diffs[0, :])),
                        len(larva_diffs[0, :])
                    ],
                    'y': [
                        0,
                        len(larva_diffs[0, :]) -  \
                        round(0.05 * len(larva_diffs[0, :]))
                    ],
                    'mode': 'lines',
                    'fill': None,
                    'line': {'width': .1, 'color': '#43d86b'},
                    'name': 'Lower bound',
                },
                {
                    'x': [
                        -round(0.05 * len(larva_diffs[0, :])),
                        len(larva_diffs[0, :])
                    ],
                    'y': [
                        0,
                        len(larva_diffs[0, :]) +  \
                        round(0.05 * len(larva_diffs[0, :]))
                    ],
                    'mode': 'lines',
                    'fill': 'tonexty',
                    'line': {'width': .1, 'color': '#c0c0c0'},
                    'name': 'Upper bound',
                },
                {
                    'x': [0, len(larva_diffs[0, :])],
                    'y': [0, len(larva_diffs[0, :])],
                    'mode': 'lines',
                    'line': {'width': .5, 'color': '#000000'},
                    'name': 'Auto = Manual',
                },
            ] + data_list + [
                {
                    'x': [auto_evals[well_idx]],
                    'y': [manual_evals[well_idx]],
                    'text': str(well_idx),
                    'mode': 'markers',
                    'marker': {'size': 10, 'color': '#ff0000'},
                    'name': 'Selected well',
                },
            ],
            'layout': {
                'annotations': [
                    {
                        'x': 0.01 * len(larva_diffs.T),
                        'y': 1.0 * len(larva_diffs.T),
                        'text': 'RMS: {:.1f}'.format(rms),
                        'showarrow': False,
                        'xanchor': 'left',
                    },
                ],
                'font': {'size': 15},
                'xaxis': {
                    'title': 'Auto',
                    'tickfont': {'size': 15},
                },
                'yaxis': {
                    'title': 'Manual',
                    'tickfont': {'size': 15},
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=50, r=0, b=40, t=70, pad=0),
            },
        }


@app.callback(
        Output('larva-summary', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'death':
        return {
                'display': 'none',
            }

    else:
        return {}


# ==========================================
#  Update the figure in the adult-summary.
# ==========================================
@app.callback(
        Output('adult-summary', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('adult-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('adult-weight-check', 'values'),
         Input('adult-weight-style', 'value'),
         Input('adult-smoothing-check', 'values'),
         Input('adult-window-size', 'value'),
         Input('adult-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('adult-dropdown', 'value'),
         State('larva-signal-type', 'value'),
         State('adult-signal-type', 'value')])
def callback(larva_coef, adult_coef, well_idx, midpoints,
        larva_weighting, larva_w_style, larva_smoothing, larva_w_size,
        larva_w_sigma, adult_weighting, adult_w_style, adult_smoothing,
        adult_w_size, adult_w_sigma, blacklist, data_root, env, detect,
        larva, adult, larva_signal_name, adult_signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if adult is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)):
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'original', 'eclosion.csv'))  \
        and detect == 'pupa-and-eclo':
        return {'data': []}

    # Load a manual evaluation of event timing
    if detect == 'pupa-and-eclo':
        if not os.path.exists(os.path.join(
                data_root, env, 'original', 'eclosion.csv')):
            #return {'data': []}
            non_manualdata = {'layout': {
                        'annotations': [
                        {
                        'x': 5.0,
                        'y': 2.0,
                        'text': 'Not Available',
                        'showarrow': False,
                        'xanchor': 'right',
                    },]}}
            return non_manualdata

        else:
            manual_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'eclosion.csv'),
                dtype=np.int16, delimiter=',').flatten()
            


    elif detect == 'death':
        if not os.path.exists(os.path.join(
                data_root, env, 'original', 'death.csv')):
            #return {'data': []}
            non_manualdata = {'layout': {
                        'annotations': [
                        {
                        'x': 5.0,
                        'y': 2.0,
                        'text': 'Not Available',
                        'showarrow': False,
                        'xanchor': 'right',
                    },]}}
            return non_manualdata

        else:
            manual_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'death.csv'),
                dtype=np.int16, delimiter=',').flatten()

    # Target wells will be evaluated
    exceptions = np.logical_or(blacklist['value'], manual_evals == 0)
    targets = np.logical_not(exceptions)

    # Load a group table
    group_tables = load_grouping_csv(data_root, env)


    # ----------------------------------------------------------
    #  Detect pupariation timing for detecting eclosion timing
    # ----------------------------------------------------------
    if larva is None:
        pupar_times = None

    else:
        # Load the data
        larva_diffs = np.load(os.path.join(data_root,
                env, 'inference', 'larva', larva, larva_signal_name)).T

        larva_thresh = THRESH_FUNC(larva_diffs, coef=larva_coef)

        larva_diffs = seasoning(
                larva_diffs, 'larva', detect, larva_w_size, larva_w_sigma,
                smooth=len(larva_smoothing) != 0,
                weight=len(larva_weighting) != 0,
                pupar_times=None,
                midpoints=midpoints,
                weight_style=larva_w_style)

        pupar_times = detect_event(larva_diffs, larva_thresh, 'larva', detect)


    # ----------------------------------------
    #  Detection of eclosion or death timing
    # ----------------------------------------
    # Load the data
    adult_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)).T

    adult_thresh = THRESH_FUNC(adult_diffs, coef=adult_coef)

    adult_diffs = seasoning(
            adult_diffs, 'adult', detect, adult_w_size, adult_w_sigma,
            smooth=len(adult_smoothing) != 0,
            weight=len(adult_weighting) != 0,
            pupar_times=pupar_times,
            midpoints=midpoints,
            weight_style=adult_w_style)

    auto_evals = detect_event(adult_diffs, adult_thresh, 'adult', detect)

    # Calculate how many frames auto-evaluation is far from manual's one
    errors = auto_evals[targets] - manual_evals[targets]

    # Calculate the root mean square
    rms = np.sqrt((errors**2).sum() / len(errors))

    # Create data points
    if group_tables == []:

        data_list = [
                {
                    'x': list(auto_evals[exceptions]),
                    'y': list(manual_evals[exceptions]),
                    'text': [str(i) for i in np.where(exceptions)[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#000000'},
                    'name': 'Exception',
                },
                {
                    'x': list(auto_evals[targets]),
                    'y': list(manual_evals[targets]),
                    'text': [str(i) for i in np.where(targets)[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#1f77b4'},
                },
            ]

    else:

        data_list = []

        for group_idx, group_table in enumerate(group_tables):

            data_list.append(
                {
                    'x': list(
                        auto_evals[np.logical_and(targets, group_table)]),
                    'y': list(
                        manual_evals[np.logical_and(targets, group_table)]),
                    'text': [str(i)
                        for i in np.where(
                            np.logical_and(targets, group_table))[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': GROUP_COLORS[group_idx]},
                    'name': 'Group{}'.format(group_idx + 1),
                })

            data_list.append(
                {
                    'x': list(
                        auto_evals[np.logical_and(exceptions, group_table)]),
                    'y': list(
                        manual_evals[np.logical_and(exceptions, group_table)]),
                    'text': [str(i)
                        for i in np.where(
                            np.logical_and(exceptions, group_table))[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#000000'},
                    'name': 'Group{}<br>Exceptions'.format(group_idx + 1),
                })


    return {
            'data': [
                {
                    'x': [
                        round(0.05 * len(adult_diffs[0, :])),
                        len(adult_diffs[0, :])
                    ],
                    'y': [
                        0,
                        len(adult_diffs[0, :]) -  \
                        round(0.05 * len(adult_diffs[0, :]))
                    ],
                    'mode': 'lines',
                    'fill': None,
                    'line': {'width': .1, 'color': '#43d86b'},
                    'name': 'Lower bound',
                },
                {
                    'x': [
                        -round(0.05 * len(adult_diffs[0, :])),
                        len(adult_diffs[0, :])
                    ],
                    'y': [
                        0,
                        len(adult_diffs[0, :]) +  \
                        round(0.05 * len(adult_diffs[0, :]))
                    ],
                    'mode': 'lines',
                    'fill': 'tonexty',
                    'line': {'width': .1, 'color': '#c0c0c0'},
                    'name': 'Upper bound',
                },
                {
                    'x': [0, len(adult_diffs[0, :])],
                    'y': [0, len(adult_diffs[0, :])],
                    'mode': 'lines',
                    'line': {'width': .5, 'color': '#000000'},
                    'name': 'Auto = Manual',
                },
            ] + data_list + [
                {
                    'x': [auto_evals[well_idx]],
                    'y': [manual_evals[well_idx]],
                    'text': str(well_idx),
                    'mode': 'markers',
                    'marker': {'size': 10, 'color': '#ff0000'},
                    'name': 'Selected well',
                },
            ],
            'layout': {
                'annotations': [
                    {
                        'x': 0.01 * len(adult_diffs.T),
                        'y': 1.0 * len(adult_diffs.T),
                        'text': 'RMS: {:.1f}'.format(rms),
                        'showarrow': False,
                        'xanchor': 'left',
                    },
                ],
                'font': {'size': 15},
                'xaxis': {
                    'title': 'Auto',
                    'tickfont': {'size': 15},
                },
                'yaxis': {
                    'title': 'Manual',
                    'tickfont': {'size': 15},
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=50, r=0, b=40, t=70, pad=0),
            },
        }


@app.callback(
        Output('adult-summary', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'death':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    else:
        return {}


# =======================================
#  Update the figure in the larva-hist.
# =======================================
@app.callback(
        Output('larva-hist', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('larva-signal-type', 'value')])
def callback(coef, well_idx, midpoints, weight, style, checks, size, sigma,
        blacklist, data_root, env, detect, larva, signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if larva is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)):
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'original', 'pupariation.csv')):
        return {'data': []}

    # Load a manual evaluation of event timing
    if not os.path.exists(os.path.join(
                data_root, env, 'original', 'pupariation.csv')):
            non_manualdata = {'layout': {
                        'annotations': [
                        {
                        'x': 5.0,
                        'y': 2.0,
                        'text': 'Not Available',
                        'showarrow': False,
                        'xanchor': 'right',
                    },]}}
            return non_manualdata

    manual_evals = np.loadtxt(
            os.path.join(data_root, env, 'original', 'pupariation.csv'),
            dtype=np.int16, delimiter=',').flatten()

    # Target wells will be evaluated
    exceptions = np.logical_or(blacklist['value'], manual_evals == 0)
    targets = np.logical_not(exceptions)

    # Load the data
    larva_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)).T

    thresholds = THRESH_FUNC(larva_diffs, coef=coef)

    larva_diffs = seasoning(
            larva_diffs, 'larva', detect, size, sigma,
            smooth=len(checks) != 0,
            weight=len(weight) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=style)

    auto_evals = detect_event(larva_diffs, thresholds, 'larva', detect)

    # Calculate how many frames auto-evaluation is far from manual's one
    errors = auto_evals - manual_evals
    errors = errors[targets]
    ns, bins = np.histogram(errors, 1000)

    # Calculate the number of inconsistent wells
    tmp = np.bincount(abs(errors))
    n_consist_5percent = tmp[:round(0.05 * larva_diffs.shape[1])].sum()
    n_consist_1percent = tmp[:round(0.01 * larva_diffs.shape[1])].sum()
    n_consist_10frames = tmp[:11].sum()

    return {
            'data': [
                {
                    'x': [
                        -round(0.05 * larva_diffs.shape[1]),
                        round(0.05 * larva_diffs.shape[1])
                    ],
                    'y': [ns.max(), ns.max()],
                    'mode': 'lines',
                    'fill': 'tozeroy',
                    'line': {'width': 0, 'color': '#c0c0c0'},
                },
                {
                    'x': list(bins[1:]),
                    'y': list(ns),
                    'mode': 'markers',
                    'type': 'bar',
                    'marker': {'size': 5, 'color': '#1f77b4'},
                },
            ],
            'layout': {
                'annotations': [
                    {
                        'x': 0.9 * larva_diffs.shape[1],
                        'y': 1.0 * ns.max(),
                        'text': '#frames: consistency',
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                    {
                        'x': 0.9 * larva_diffs.shape[1],
                        'y': 0.9 * ns.max(),
                        'text': '{} (5%): {:.1f}% ({}/{})'.format(
                            round(0.05 * larva_diffs.shape[1]),
                            100 * n_consist_5percent / targets.sum(),
                            n_consist_5percent,
                            targets.sum()),
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                    {
                        'x': 0.9 * larva_diffs.shape[1],
                        'y': 0.8 * ns.max(),
                        'text': '{} (1%): {:.1f}% ({}/{})'.format(
                            round(0.01 * larva_diffs.shape[1]),
                            100 * n_consist_1percent / targets.sum(),
                            n_consist_1percent,
                            targets.sum()),
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                    {
                        'x': 0.9 * larva_diffs.shape[1],
                        'y': 0.7 * ns.max(),
                        'text': '10: {:.1f}% ({}/{})'.format(
                            100 * n_consist_10frames / targets.sum(),
                            n_consist_10frames,
                            targets.sum()),
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                ],
                'font': {'size': 15},
                'xaxis': {
                    'title': 'auto - manual',
                    'range': [-len(larva_diffs.T), len(larva_diffs.T)],
                    'tickfont': {'size': 15},
                },
                'yaxis': {
                    'title': 'Count',
                    'tickfont': {'size': 15},
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=50, r=0, b=50, t=70, pad=0),
            },
        }


@app.callback(
        Output('larva-hist', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'death':
        return {
                'display': 'none',
            }

    else:
        return {}


# =======================================
#  Update the figure in the adult-hist.
# =======================================
@app.callback(
        Output('adult-hist', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('adult-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('adult-weight-check', 'values'),
         Input('adult-weight-style', 'value'),
         Input('adult-smoothing-check', 'values'),
         Input('adult-window-size', 'value'),
         Input('adult-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('adult-dropdown', 'value'),
         State('larva-signal-type', 'value'),
         State('adult-signal-type', 'value')])
def callback(larva_coef, adult_coef, well_idx, midpoints,
        larva_weighting, larva_w_style, larva_smoothing, larva_w_size,
        larva_w_sigma, adult_weighting, adult_w_style, adult_smoothing,
        adult_w_size, adult_w_sigma, blacklist, data_root, env, detect,
        larva, adult, larva_signal_name, adult_signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if adult is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)):
        return {'data': []}

    # Load a manual evaluation of event timing
    if detect == 'pupa-and-eclo':
        if not os.path.exists(os.path.join(
                data_root, env, 'original', 'eclosion.csv')):
            #return {'data': []}
            non_manualdata = {'layout': {
                        'annotations': [
                        {
                        'x': 5.0,
                        'y': 2.0,
                        'text': 'Not Available',
                        'showarrow': False,
                        'xanchor': 'right',
                    },]}}
            return non_manualdata

        manual_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'eclosion.csv'),
                dtype=np.int16, delimiter=',').flatten()

    elif detect == 'death':
        if not os.path.exists(os.path.join(
                data_root, env, 'original', 'death.csv')):
            non_manualdata = {'layout': {
                        'annotations': [
                        {
                        'x': 5.0,
                        'y': 2.0,
                        'text': 'Not Available',
                        'showarrow': False,
                        'xanchor': 'right',
                    },]}}
            return non_manualdata

        manual_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'death.csv'),
                dtype=np.int16, delimiter=',').flatten()

    # Target wells will be evaluated
    exceptions = np.logical_or(blacklist['value'], manual_evals == 0)
    targets = np.logical_not(exceptions)


    # ----------------------------------------------------------
    #  Detect pupariation timing for detecting eclosion timing
    # ----------------------------------------------------------
    if larva is None:
        pupar_times = None

    else:
        # Load the data
        larva_diffs = np.load(os.path.join(data_root,
                env, 'inference', 'larva', larva, larva_signal_name)).T

        larva_thresh = THRESH_FUNC(larva_diffs, coef=larva_coef)

        larva_diffs = seasoning(
                larva_diffs, 'larva', detect, larva_w_size, larva_w_sigma,
                smooth=len(larva_smoothing) != 0,
                weight=len(larva_weighting) != 0,
                pupar_times=None,
                midpoints=midpoints,
                weight_style=larva_w_style)

        pupar_times = detect_event(larva_diffs, larva_thresh, 'larva', detect)


    # ----------------------------------------
    #  Detection of eclosion or death timing
    # ----------------------------------------
    # Load the data
    adult_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)).T

    adult_thresh = THRESH_FUNC(adult_diffs, coef=adult_coef)

    adult_diffs = seasoning(
            adult_diffs, 'adult', detect, adult_w_size, adult_w_sigma,
            smooth=len(adult_smoothing) != 0,
            weight=len(adult_weighting) != 0,
            pupar_times=pupar_times,
            midpoints=midpoints,
            weight_style=adult_w_style)

    auto_evals = detect_event(adult_diffs, adult_thresh, 'adult', detect)

    # Calculate how many frames auto-evaluation is far from manual's one
    errors = auto_evals - manual_evals
    errors = errors[targets]
    ns, bins = np.histogram(errors, 1000)

    # Calculate the number of inconsistent wells
    tmp = np.bincount(abs(errors))
    n_consist_5percent = tmp[:round(0.05 * adult_diffs.shape[1])].sum()
    n_consist_1percent = tmp[:round(0.01 * adult_diffs.shape[1])].sum()
    n_consist_10frames = tmp[:11].sum()

    return {
            'data': [
                {
                    'x': [
                        -round(0.05 * adult_diffs.shape[1]),
                        round(0.05 * adult_diffs.shape[1])
                    ],
                    'y': [ns.max(), ns.max()],
                    'mode': 'lines',
                    'fill': 'tozeroy',
                    'line': {'width': 0, 'color': '#c0c0c0'},
                },
                {
                    'x': list(bins[1:]),
                    'y': list(ns),
                    'mode': 'markers',
                    'type': 'bar',
                    'marker': {'size': 5, 'color': '#1f77b4'},
                },
            ],
            'layout': {
                'annotations': [
                    {
                        'x': 0.9 * adult_diffs.shape[1],
                        'y': 1.0 * ns.max(),
                        'text': '#frames: consistency',
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                    {
                        'x': 0.9 * adult_diffs.shape[1],
                        'y': 0.9 * ns.max(),
                        'text': '{} (5%): {:.1f}% ({}/{})'.format(
                            round(0.05 * adult_diffs.shape[1]),
                            100 * n_consist_5percent / targets.sum(),
                            n_consist_5percent,
                            targets.sum()),
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                    {
                        'x': 0.9 * adult_diffs.shape[1],
                        'y': 0.8 * ns.max(),
                        'text': '{} (1%): {:.1f}% ({}/{})'.format(
                            round(0.01 * adult_diffs.shape[1]),
                            100 * n_consist_1percent / targets.sum(),
                            n_consist_1percent,
                            targets.sum()),
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                    {
                        'x': 0.9 * adult_diffs.shape[1],
                        'y': 0.7 * ns.max(),
                        'text': '10: {:.1f}% ({}/{})'.format(
                            100 * n_consist_10frames / targets.sum(),
                            n_consist_10frames,
                            targets.sum()),
                        'showarrow': False,
                        'xanchor': 'right',
                    },
                ],
                'font': {'size': 15},
                'xaxis': {
                    'title': 'auto - manual',
                    'range': [-len(adult_diffs.T), len(adult_diffs.T)],
                    'tickfont': {'size': 15},
                },
                'yaxis': {
                    'title': 'Count',
                    'tickfont': {'size': 15},
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=50, r=0, b=50, t=70, pad=0),
            },
        }


@app.callback(
        Output('adult-hist', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}

    if detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'death':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    else:
        return {}


# ==============================================
#  Update the figure in the pupa-vs-eclo plot.
# ==============================================
@app.callback(
        Output('pupa-vs-eclo', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('adult-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('adult-weight-check', 'values'),
         Input('adult-weight-style', 'value'),
         Input('adult-smoothing-check', 'values'),
         Input('adult-window-size', 'value'),
         Input('adult-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('adult-dropdown', 'value'),
         State('larva-signal-type', 'value'),
         State('adult-signal-type', 'value')])
def callback(larva_coef, adult_coef, well_idx, midpoints, larva_weighting,
        larva_w_style, larva_smoothing, larva_w_size, larva_w_sigma,
        adult_weighting, adult_w_style, adult_smoothing, adult_w_size,
        adult_w_sigma, blacklist, data_root, env, detect, larva, adult, larva_signal_name, adult_signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if larva is None:
        return {'data': []}
    if adult is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'larva', larva, larva_signal_name)):
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)):
        return {'data': []}
    if detect == 'death':
        return {'data': []}

    # Load a blacklist and whitelist
    blacklist = np.array(blacklist['value'])
    whitelist = np.logical_not(blacklist)

    # Load the data
    larva_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'larva', larva, larva_signal_name)).T

    larva_thresh = THRESH_FUNC(larva_diffs, coef=larva_coef)

    larva_diffs = seasoning(
            larva_diffs, 'larva', detect, larva_w_size, larva_w_sigma,
            smooth=len(larva_smoothing) != 0,
            weight=len(larva_weighting) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=larva_w_style)

    # Evaluate event timing
    # Compute event times from signals
    # Scan the signal from the right hand side.
    pupars = (larva_diffs.shape[1]
            - (np.fliplr(larva_diffs) > larva_thresh).argmax(axis=1))
    pupars[pupars == larva_diffs.shape[1]] = 0

    adult_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)).T

    adult_thresh = THRESH_FUNC(adult_diffs, coef=adult_coef)

    adult_diffs = seasoning(
            adult_diffs, 'adult', detect, adult_w_size, adult_w_sigma,
            smooth=len(adult_smoothing) != 0,
            weight=len(adult_weighting) != 0,
            pupar_times=pupars,
            midpoints=midpoints,
            weight_style=adult_w_style)

    eclos = (adult_diffs > adult_thresh).argmax(axis=1)

    return {
            'data': [
                {
                    'x': list(pupars[blacklist]),
                    'y': list(eclos[blacklist]),
                    'text': [str(i) for i in np.where(blacklist)[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#000000'},
                    'name': 'Well in Blacklist',
                },
                {
                    'x': list(pupars[whitelist]),
                    'y': list(eclos[whitelist]),
                    'text': [str(i) for i in np.where(whitelist)[0]],
                    'mode': 'markers',
                    'marker': {'size': 4, 'color': '#1f77b4'},
                    'name': 'Well in Whitelist',
                },
                {
                    'x': [pupars[well_idx]],
                    'y': [eclos[well_idx]],
                    'text': str(well_idx),
                    'mode': 'markers',
                    'marker': {'size': 10, 'color': '#ff0000'},
                    'name': 'Selected well',
                },
            ],
            'layout': {
                'font': {'size': 15},
                'xaxis': {
                    'title': 'Pupariation',
                    'tickfont': {'size': 15},
                    'range': [
                        -0.1 * len(larva_diffs.T),
                        1.1 * len(larva_diffs.T)],
                },
                'yaxis': {
                    'title': 'Eclosion',
                    'tickfont': {'size': 15},
                    'range': [
                        -0.1 * len(adult_diffs.T),
                        1.1 * len(adult_diffs.T)],
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=50, r=0, b=50, t=70, pad=0),
            },
        }


@app.callback(
        Output('pupa-vs-eclo', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'death':
        return {'display': 'none'}

    else:
        return {}


# ===========================================
#  Update the figure in the survival-curve.
# ===========================================
@app.callback(
        Output('survival-curve', 'figure'),
        [Input('adult-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('adult-weight-check', 'values'),
         Input('adult-weight-style', 'value'),
         Input('adult-smoothing-check', 'values'),
         Input('adult-window-size', 'value'),
         Input('adult-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('adult-dropdown', 'value'),
         State('adult-signal-type', 'value')])
def callback(coef, well_idx, midpoints, weight, style, checks, size, sigma,
        blacklist, data_root, env, detect, adult, signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if adult is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'adult', adult, signal_name)):
        return {'data': []}
    if detect == 'pupariation':
        return {'data': []}
    if detect == 'pupa-and-eclo':
        return {'data': []}

    # Load a blacklist and whitelist
    whitelist = np.logical_not(np.array(blacklist['value']))

    # Load a group table
    group_tables = load_grouping_csv(data_root, env)

    # Load the data
    adult_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'adult', adult, signal_name)).T

    thresholds = THRESH_FUNC(adult_diffs, coef=coef)

    adult_diffs = seasoning(
            adult_diffs, 'adult', detect, size, sigma,
            smooth=len(checks) != 0,
            weight=len(weight) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=style)

    auto_evals = detect_event(adult_diffs, thresholds, 'adult', detect)

    if group_tables == []:
        # Compute survival ratio of all the animals
        survival_ratio = np.zeros_like(adult_diffs)

        for well_idx, event_time in enumerate(auto_evals):

            survival_ratio[well_idx, :event_time] = 1

        survival_ratio =  \
                100 * survival_ratio[whitelist].sum(axis=0) /  \
                len(survival_ratio[whitelist])

        data_list = [
            {
                'x': list(range(len(survival_ratio))),
                'y': list(survival_ratio),
                'mode': 'lines',
                'line': {'size': 2, 'color': '#ff4500'},
                'name': 'Group1'
            }]

    else:
        survival_ratios = []
        for group_idx, group_table in enumerate(group_tables):

            survival_ratio = np.zeros_like(adult_diffs)

            for well_idx, (event_time, in_group)  \
                    in enumerate(zip(auto_evals, group_table)):

                survival_ratio[well_idx, :event_time] = in_group

            survival_ratio =  \
                    100 * survival_ratio[whitelist].sum(axis=0) /  \
                    group_table[whitelist].sum()

            survival_ratios.append(survival_ratio)
    
        data_list =[]
        for group_idx, (group_table, survival_ratio)  \
                in enumerate(zip(group_tables, survival_ratios)):

            data_list.append(
                {
                    'x': list(range(len(survival_ratio))),
                    'y': list(survival_ratio),
                    'mode': 'lines',
                    'marker': {'size': 2, 'color': GROUP_COLORS[group_idx]},
                    'name': 'Group{}'.format(group_idx + 1),
                })

    return {
            'data': data_list + [
                {
                    'x': [0, len(survival_ratio)],
                    'y': [100, 100],
                    'mode': 'lines',
                    'line': {'width': 1, 'color': '#000000'},
                },
                ],
            'layout': {
                'font': {'size': 15},
                'xaxis': {
                    'title': 'Frame',
                    'range': [0, 1.1 * len(survival_ratio)],
                    'tickfont': {'size': 15},
                },
                'yaxis': {
                    'title': 'Survival Ratio [%]',
                    'tickfont': {'size': 15},
                    'range': [0, 110],
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=50, r=0, b=50, t=70, pad=0),
            },
        }


@app.callback(
        Output('survival-curve', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}

    elif detect == 'pupa-and-eclo':
        return {'display': 'none'}

    elif detect == 'death':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    else:
        return {}


# ===========================================
#  Update the boxplot for larva
# ===========================================
@app.callback(
        Output('larva-boxplot', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('larva-signal-type', 'value')])
def callback(coef, well_idx, midpoints, weight, style, checks, size, sigma,
        blacklist, data_root, env, detect, larva, signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if larva is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)):
        return {'data': []}
    if detect == 'death':
        return {'data': []}

    # Load a blacklist and whitelist
    whitelist = np.logical_not(np.array(blacklist['value']))

    # Load a group table
    group_tables = load_grouping_csv(data_root, env)

    # Load the data
    larva_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)).T

    larva_diffs = seasoning(
            larva_diffs, 'larva', detect, size, sigma,
            smooth=len(checks) != 0,
            weight=len(weight) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=style)

    # Compute thresholds
    thresholds = my_threshold.entire_stats(larva_diffs, coef=coef)

    auto_evals = detect_event(larva_diffs, thresholds, 'larva', detect)

    # Make data to be drawn
    if group_tables == []:
        data = []
        data.append(
                go.Box(
                    x=list(auto_evals[whitelist]),
                    name='Group1',
                    boxpoints='all',
                    pointpos=1.8,
                    marker={'size': 2},
                    line={'width': 2},
                    text=[str(i) for i in np.where(whitelist)[0]],
                    boxmean='sd',
                )
            )

    else :
        data =[]
        for group_idx, group_table in enumerate(group_tables):
            data.append(
                go.Box(
                    x=list(auto_evals[np.logical_and(whitelist, group_table)]),
                    name='Group{}'.format(group_idx +1),
                    boxpoints='all',
                    pointpos=1.8,
                    marker={'size': 2, 'color': GROUP_COLORS[group_idx]},
                    line={'width': 2, 'color': GROUP_COLORS[group_idx]},
                    text=[str(i)
                        for i in np.where(
                            np.logical_and(whitelist, group_table))[0]],
                    boxmean='sd',
                )
            )

    return {
            'data': data,
            'layout': {
                'font': {'size': 15},
                'xaxis': {
                    'title': 'Frame',
                    'tickfont': {'size': 15},
                    'range': [0, 1.1 * len(larva_diffs.T)],
                },
                'yaxis': {
                    'tickfont': {'size': 15},
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=70, r=0, b=50, t=70, pad=0),
            },
        }


@app.callback(
        Output('larva-boxplot', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'death':
        return {'display': 'none',
            }

    else:
        return {}


# ===========================================
#  Update the boxplot for adult
# ===========================================
@app.callback(
        Output('adult-boxplot', 'figure'),
        [Input('larva-thresh-selector', 'value'),
         Input('adult-thresh-selector', 'value'),
         Input('well-selector', 'value'),
         Input('hidden-midpoint', 'data'),
         Input('larva-weight-check', 'values'),
         Input('larva-weight-style', 'value'),
         Input('larva-smoothing-check', 'values'),
         Input('larva-window-size', 'value'),
         Input('larva-window-sigma', 'value'),
         Input('adult-weight-check', 'values'),
         Input('adult-weight-style', 'value'),
         Input('adult-smoothing-check', 'values'),
         Input('adult-window-size', 'value'),
         Input('adult-window-sigma', 'value'),
         Input('hidden-blacklist', 'data')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('adult-dropdown', 'value'),
         State('larva-signal-type', 'value'),
         State('adult-signal-type', 'value')])
def callback(larva_coef, adult_coef, well_idx, midpoints,
        larva_weighting, larva_w_style, larva_smoothing, larva_w_size,
        larva_w_sigma, adult_weighting, adult_w_style, adult_smoothing,
        adult_w_size, adult_w_sigma, blacklist, data_root, env, detect,
        larva, adult, larva_signal_name, adult_signal_name):
    # Guard
    if env is None:
        return {'data': []}
    if adult is None:
        return {'data': []}
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)):
        return {'data': []}
    if detect == 'pupariation':
        return {'data': []}

    # Load a blacklist and whitelist
    whitelist = np.logical_not(np.array(blacklist['value']))

    # Load a group table
    group_tables = load_grouping_csv(data_root, env)


    # ----------------------------------------------------------
    #  Detect pupariation timing for detecting eclosion timing
    # ----------------------------------------------------------
    if larva is None:
        pupar_times = None

    else:
        # Load the data
        larva_diffs = np.load(os.path.join(data_root,
                env, 'inference', 'larva', larva, larva_signal_name)).T

        larva_thresh = THRESH_FUNC(larva_diffs, coef=larva_coef)

        larva_diffs = seasoning(
                larva_diffs, 'larva', detect, larva_w_size, larva_w_sigma,
                smooth=len(larva_smoothing) != 0,
                weight=len(larva_weighting) != 0,
                pupar_times=None,
                midpoints=midpoints,
                weight_style=larva_w_style)

        pupar_times = detect_event(larva_diffs, larva_thresh, 'larva', detect)


    # ----------------------------------------
    #  Detection of eclosion or death timing
    # ----------------------------------------
    # Load the data
    adult_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)).T

    adult_thresh = THRESH_FUNC(adult_diffs, coef=adult_coef)

    adult_diffs = seasoning(
            adult_diffs, 'adult', detect, adult_w_size, adult_w_sigma,
            smooth=len(adult_smoothing) != 0,
            weight=len(adult_weighting) != 0,
            pupar_times=pupar_times,
            midpoints=midpoints,
            weight_style=adult_w_style)

    auto_evals = detect_event(adult_diffs, adult_thresh, 'adult', detect)

    # Make data to be drawn
    if group_tables == []:
        data = []
        data.append(
                go.Box(
                    x=list(auto_evals[whitelist]),
                    name='Group1',
                    boxpoints='all',
                    pointpos=1.8,
                    marker={'size': 2},
                    line={'width': 2},
                    text=[str(i) for i in np.where(whitelist)[0]],
                    boxmean='sd',
                )
            )

    else :
        data =[]
        for group_idx, group_table in enumerate(group_tables):
            data.append(
                go.Box(
                    x=list(auto_evals[np.logical_and(whitelist, group_table)]),
                    name='Group{}'.format(group_idx +1),
                    boxpoints='all',
                    pointpos=1.8,
                    marker={'size': 2, 'color': GROUP_COLORS[group_idx]},
                    line={'width': 2, 'color': GROUP_COLORS[group_idx]},
                    text=[str(i)
                        for i in np.where(
                            np.logical_and(whitelist, group_table))[0]],
                    boxmean='sd',
                )
            )

    return {
            'data': data,
            'layout': {
                'font': {'size': 15},
                'xaxis': {
                    'title': 'Frame',
                    'tickfont': {'size': 15},
                    'range': [0, 1.1 * len(adult_diffs.T)],
                },
                'yaxis': {
                    'tickfont': {'size': 15},
                },
                'showlegend': False,
                'hovermode': 'closest',
                'margin': go.layout.Margin(l=70, r=0, b=50, t=70, pad=0),
            },
        }


@app.callback(
        Output('adult-boxplot', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    elif detect == 'death':
        return {
                'display': 'inline-block',
                'height': '300px',
                'width': '20%',
            }

    else:
        return {}


# =======================================================================
#  Store image file names and their timestamps as json in a hidden div.
# =======================================================================
@app.callback(
        Output('hidden-timestamp', 'data'),
        [Input('env-dropdown', 'value')],
        [State('data-root', 'children')])
def callback(env, data_root):
    # Guard
    if env is None:
        return

    # Load an original image
    orgimg_paths = sorted(glob.glob(
            os.path.join(data_root, env, 'original', '*.jpg')))

    return {
            'Image name': [os.path.basename(path) for path in orgimg_paths],
            'Create time': [get_create_time(path) for path in orgimg_paths],
        }


def get_create_time(path):
    DateTimeDigitized = PIL.Image.open(path)._getexif()[36868]
    # '2016:02:05 17:20:53' -> '2016-02-05 17:20:53'
    DateTimeDigitized = DateTimeDigitized[:4] + '-' + DateTimeDigitized[5:]
    DateTimeDigitized = DateTimeDigitized[:7] + '-' + DateTimeDigitized[8:]

    return DateTimeDigitized


# ======================================================
#  Timestamp table
# ======================================================
@app.callback(
        Output('timestamp-table', 'children'),
        [Input('tabs', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('hidden-timestamp', 'data')])
def callback(tab_name, data_root, env, timestamps):
    # Guard
    if data_root is None:
        return 'Not available.'
    if env is None:
        return 'Not available.'
    if timestamps is None:
        return 'Now loading...'
    if tab_name != 'tab-2':
        return

    df = pd.DataFrame([timestamps['Image name'], timestamps['Create time']],
            index=['Image name', 'Create time']).T

    data = [
            {'Image name': image_name, 'Create time': create_time}
            for image_name, create_time in zip(
                    timestamps['Image name'], timestamps['Create time'])]

    return [
            html.H4('Timestamp'),
            dash_table.DataTable(
                columns=[
                    {'id': 'Image name', 'name': 'Image name'},
                    {'id': 'Create time', 'name': 'Create time'}],
                data=data,
                n_fixed_rows=1,
                style_table={'width': '100%'},
                pagination_mode=False,
            ),
            html.A(
                'Download the Data',
                id='download-link',
                download='Timestamp({}).csv'.format(env[0:20]),
                href='data:text/csv;charset=utf-8,' + df.to_csv(index=False),
                target='_blank',
            ),
        ]


# ======================================================
#  Manual table for larva
# ======================================================
@app.callback(
        Output('larva-man-table', 'children'),
        [Input('tabs', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value')])
def callback(tab_name, data_root, env, detect, larva):
    # Guard
    if data_root is None:
        return 'Not available.'
    if env is None:
        return 'Not available.'
    if detect is None:
        return 'Not available.'
    if tab_name != 'tab-2':
        return

    # Load a mask params
    with open(os.path.join(data_root, env, 'mask_params.json')) as f:
        params = json.load(f)

    if detect == 'pupariation' or detect == 'pupa-and-eclo':
        if not os.path.exists(os.path.join(
                data_root, env, 'original', 'pupariation.csv')):
            return 'Not available.'

        # Load a manual data
        larva_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'pupariation.csv'),
                dtype=np.int16, delimiter=',').flatten()

        larva_evals = larva_evals.reshape(
                params['n-rows']*params['n-plates'], params['n-clms'])

        larva_csv = \
                  'data:text/csv;charset=utf-8,'  \
                + 'Dataset,{}\n'.format(env)  \
                + 'Morphology,larva\n'  \
                + 'Inference Data,{}\n'.format(larva)  \
                + 'Event Timing\n'  \
                + pd.DataFrame(larva_evals).to_csv(
                        index=False, encoding='utf-8', header=False)

        larva_style = [{
                'if': {
                    'column_id': '{}'.format(clm),
                    'filter': 'num({2}) > {0} && {1} >= num({3})'.format(
                            clm, clm, int(t)+100, int(t)),
                },
                'backgroundColor': '#{:02X}{:02X}00'.format(int(c), int(c)),
                'color': 'black',
            }
            for clm in range(params['n-clms'])
            for t, c in zip(
                range(0, larva_evals.max(), 100),
                np.linspace(0, 255, len(range(0, larva_evals.max(), 100))))
        ]

        return [
                html.H4('Event Timings of Larva (manual)'),
                dash_table.DataTable(
                    columns=[{'name': str(clm), 'id': str(clm)}
                            for clm in range(params['n-clms'])],
                    data=pd.DataFrame(larva_evals).to_dict('rows'),
                    style_data_conditional=larva_style,
                    style_table={'width': '100%'}
                ),
                html.A(
                    'Download the Data',
                    id='download-link',
                    download='Manual_Detection.csv',
                    href=larva_csv,
                    target='_blank',
                ),
            ]

    if detect == 'death':
        return []


@app.callback(
        Output('larva-man-table', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    elif detect == 'death':
        return {'display': 'none'}

    else:
        return {}


# ======================================================
#  Auto table for larva
# ======================================================
@app.callback(
        Output('larva-auto-table', 'children'),
        [Input('tabs', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('larva-thresh-selector', 'value'),
         State('hidden-midpoint', 'data'),
         State('larva-weight-check', 'values'),
         State('larva-weight-style', 'value'),
         State('larva-window-size', 'value'),
         State('larva-window-sigma', 'value'),
         State('larva-smoothing-check', 'values'),
         State('larva-signal-type', 'value')])
def callback(tab_name, data_root, env, detect, larva, coef,
        midpoints, weight, style, size, sigma, checks, signal_name):
    # Guard
    if data_root is None:
        return 'Not available.'
    if env is None:
        return 'Not available.'
    if tab_name != 'tab-2':
        return
    if detect is None:
        return 'Not available.'
    if larva is None:
        return 'Not available.'
    if detect == 'death':
        return []
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)):
        return 'Not available.'

    # Load a mask params
    with open(os.path.join(data_root, env, 'mask_params.json')) as f:
        params = json.load(f)

    larva_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'larva', larva, signal_name)).T
    diffs_mean = larva_diffs.mean()
    diffs_std = larva_diffs.std()
    thresholds = THRESH_FUNC(larva_diffs, coef=coef)

    larva_diffs = seasoning(
            larva_diffs, 'larva', detect, size, sigma,
            smooth=len(checks) != 0,
            weight=len(weight) != 0,
            pupar_times=None,
            midpoints=midpoints,
            weight_style=style)

    auto_evals = detect_event(larva_diffs, thresholds, 'larva', detect)

    auto_evals = auto_evals.reshape(
            params['n-rows']*params['n-plates'], params['n-clms'])

    auto_to_csv =  \
              'data:text/csv;charset=utf-8,'  \
            + 'Dataset,{}\n'.format(env)  \
            + 'Morphology,larva\n'  \
            + 'Inference Data,{}\n'.format(larva)  \
            + 'Threshold Value,{}\n'.format(thresholds[0, 0])  \
            + '(Threshold Value = mean + coef * std)\n'  \
            + 'Mean (mean),{}\n'.format(diffs_mean)  \
            + 'Coefficient (coef),{}\n'.format(coef)  \
            + 'Standard Deviation (std),{}\n'.format(diffs_std)  \
            + 'Smoothing Window Size,{}\n'.format(size)  \
            + 'Smoothing Sigma,{}\nEvent Timing\n'.format(sigma)  \
            + pd.DataFrame(auto_evals).to_csv(
                    index=False, encoding='utf-8', header=False),

    style = [{
            'if': {
                'column_id': '{}'.format(clm),
                'filter': 'num({2}) > {0} && {1} >= num({3})'.format(
                        clm, clm, int(t)+100, int(t)),
            },
            'backgroundColor': '#{:02X}{:02X}00'.format(int(c), int(c)),
            'color': 'black',
        }
        for clm in range(params['n-clms'])
        for t, c in zip(
            range(0, auto_evals.max(), 100),
            np.linspace(0, 255, len(range(0, auto_evals.max(), 100))))
    ]

    return [
            html.H4('Event Timings of Larva (auto)'),
            dash_table.DataTable(
                columns=[{'name': str(clm), 'id': str(clm)}
                        for clm in range(params['n-clms'])],
                data=pd.DataFrame(auto_evals).to_dict('rows'),
                style_data_conditional=style,
                style_table={'width': '100%'}
            ),
            html.A(
                'Download the Data',
                id='download-link',
                download='Auto_Detection.csv',
                href=auto_to_csv,
                target='_blank',
            ),
        ]


@app.callback(
        Output('larva-auto-table', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    elif detect == 'death':
        return {'display': 'none'}

    else:
        return {}


# ======================================================
#  Manual table for adult
# ======================================================
@app.callback(
        Output('adult-man-table', 'children'),
        [Input('tabs', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('adult-dropdown', 'value')])
def callback(tab_name, data_root, env, detect, adult):
    # Guard
    if data_root is None:
        return 'Not available.'
    if env is None:
        return 'Not available.'
    if detect is None:
        return 'Not available.'
    if detect == 'pupariation':
        return 'Not available.'
    if tab_name != 'tab-2':
        return

    # Load a mask params
    with open(os.path.join(data_root, env, 'mask_params.json')) as f:
        params = json.load(f)

    if detect == 'pupa-and-eclo':
        if not os.path.exists(os.path.join(
                data_root, env, 'original', 'eclosion.csv')):
            return 'Not available.'

        # Load a manual data
        adult_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'eclosion.csv'),
                dtype=np.int16, delimiter=',').flatten()

    elif detect == 'death':
        if not os.path.exists(os.path.join(
                data_root, env, 'original', 'death.csv')):
            return 'Not available.'

        # Load a manual data
        adult_evals = np.loadtxt(
                os.path.join(data_root, env, 'original', 'death.csv'),
                dtype=np.int16, delimiter=',').flatten()

    adult_evals = adult_evals.reshape(
            params['n-rows']*params['n-plates'], params['n-clms'])

    adult_csv = \
              'data:text/csv;charset=utf-8,'  \
            + 'Dataset,{}\n'.format(env)  \
            + 'Morphology,adult\n'  \
            + 'Inference Data,{}\n'.format(adult)  \
            + 'Event Timing\n'  \
            + pd.DataFrame(adult_evals).to_csv(
                    index=False, encoding='utf-8', header=False)

    adult_style = [{
            'if': {
                'column_id': '{}'.format(clm),
                'filter': 'num({2}) > {0} && {1} >= num({3})'.format(
                        clm, clm, int(t)+100, int(t)),
            },
            'backgroundColor': '#{:02X}{:02X}00'.format(int(c), int(c)),
            'color': 'black',
        }
        for clm in range(params['n-clms'])
        for t, c in zip(
            range(0, adult_evals.max(), 100),
            np.linspace(0, 255, len(range(0, adult_evals.max(), 100))))
    ]

    return [
            html.H4('Event Timings of Adult (manual)'),
            dash_table.DataTable(
                columns=[{'name': str(clm), 'id': str(clm)}
                        for clm in range(params['n-clms'])],
                data=pd.DataFrame(adult_evals).to_dict('rows'),
                style_data_conditional=adult_style,
                style_table={'width': '100%'}
            ),
            html.A(
                'Download the Data',
                id='download-link',
                download='Manual_Detection.csv',
                href=adult_csv,
                target='_blank',
            ),
        ]


@app.callback(
        Output('adult-man-table', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    elif detect == 'death':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    else:
        return {}


# ======================================================
#  Auto table for adult
# ======================================================
@app.callback(
        Output('adult-auto-table', 'children'),
        [Input('tabs', 'value')],
        [State('data-root', 'children'),
         State('env-dropdown', 'value'),
         State('detect-target', 'value'),
         State('larva-dropdown', 'value'),
         State('adult-dropdown', 'value'),
         State('larva-thresh-selector', 'value'),
         State('adult-thresh-selector', 'value'),
         State('hidden-midpoint', 'data'),
         State('larva-weight-check', 'values'),
         State('larva-weight-style', 'value'),
         State('larva-window-size', 'value'),
         State('larva-window-sigma', 'value'),
         State('larva-smoothing-check', 'values'),
         State('adult-weight-check', 'values'),
         State('adult-weight-style', 'value'),
         State('adult-window-size', 'value'),
         State('adult-window-sigma', 'value'),
         State('adult-smoothing-check', 'values'),
         State('larva-signal-type', 'value'),
         State('adult-signal-type', 'value')])
def callback(tab_name, data_root, env, detect, larva, adult, larva_coef,
        adult_coef, midpoints, larva_weighting, larva_w_style, larva_w_size,
        larva_w_sigma, larva_smoothing, adult_weighting, adult_w_style,
        adult_w_size, adult_w_sigma, adult_smoothing, larva_signal_name,
        adult_signal_name):
    # Guard
    if data_root is None:
        return 'Not available.'
    if env is None:
        return 'Not available.'
    if tab_name != 'tab-2':
        return
    if detect is None:
        return 'Not available.'
    if adult is None:
        return 'Not available.'
    if not os.path.exists(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)):
        return 'Not available.'

    # Load a mask params
    with open(os.path.join(data_root, env, 'mask_params.json')) as f:
        params = json.load(f)


    # ----------------------------------------------------------
    #  Detect pupariation timing for detecting eclosion timing
    # ----------------------------------------------------------
    if larva is None:
        pupar_times = None

    else:
        larva_diffs = np.load(os.path.join(data_root,
                env, 'inference', 'larva', larva, larva_signal_name)).T

        larva_thresh = THRESH_FUNC(larva_diffs, coef=larva_coef)

        larva_diffs = seasoning(
                larva_diffs, 'larva', detect, larva_w_size, larva_w_sigma,
                smooth=len(larva_smoothing) != 0,
                weight=len(larva_weighting) != 0,
                pupar_times=None,
                midpoints=midpoints,
                weight_style=larva_w_style)

        pupar_times = detect_event(larva_diffs, larva_thresh, 'larva', detect)


    # ----------------------------------------
    #  Detection of eclosion or death timing
    # ----------------------------------------
    # Load the data
    adult_diffs = np.load(os.path.join(
            data_root, env, 'inference', 'adult', adult, adult_signal_name)).T
    diffs_mean = adult_diffs.mean()
    diffs_std = adult_diffs.std()
    adult_thresh = THRESH_FUNC(adult_diffs, coef=adult_coef)

    adult_diffs = seasoning(
            adult_diffs, 'adult', detect, adult_w_size, adult_w_sigma,
            smooth=len(adult_smoothing) != 0,
            weight=len(adult_weighting) != 0,
            pupar_times=pupar_times,
            midpoints=midpoints,
            weight_style=adult_w_style)

    auto_evals = detect_event(adult_diffs, adult_thresh, 'adult', detect)

    auto_evals = auto_evals.reshape(
            params['n-rows']*params['n-plates'], params['n-clms'])

    auto_to_csv =  \
              'data:text/csv;charset=utf-8,'  \
            + 'Dataset,{}\n'.format(env)  \
            + 'Morphology,adult\n'  \
            + 'Inference Data,{}\n'.format(adult)  \
            + 'Threshold Value,{}\n'.format(adult_thresh[0, 0])  \
            + '(Threshold Value = mean + coef * std)\n'  \
            + 'Mean (mean),{}\n'.format(diffs_mean)  \
            + 'Coefficient (coef),{}\n'.format(adult_coef)  \
            + 'Standard Deviation (std),{}\n'.format(diffs_std)  \
            + 'Smoothing Window Size,{}\n'.format(adult_w_size)  \
            + 'Smoothing Sigma,{}\nEvent Timing\n'.format(adult_w_sigma)  \
            + pd.DataFrame(auto_evals).to_csv(
                    index=False, encoding='utf-8', header=False),

    style = [{
            'if': {
                'column_id': '{}'.format(clm),
                'filter': 'num({2}) > {0} && {1} >= num({3})'.format(
                        clm, clm, int(t)+100, int(t)),
            },
            'backgroundColor': '#{:02X}{:02X}00'.format(int(c), int(c)),
            'color': 'black',
        }
        for clm in range(params['n-clms'])
        for t, c in zip(
            range(0, auto_evals.max(), 100),
            np.linspace(0, 255, len(range(0, auto_evals.max(), 100))))
    ]

    return [
            html.H4('Event Timings of Adult (auto)'),
            dash_table.DataTable(
                columns=[{'name': str(clm), 'id': str(clm)}
                        for clm in range(params['n-clms'])],
                data=pd.DataFrame(auto_evals).to_dict('rows'),
                style_data_conditional=style,
                style_table={'width': '100%'}
            ),
            html.A(
                'Download the Data',
                id='download-link',
                download='Auto_Detection.csv',
                href=auto_to_csv,
                target='_blank',
            ),
        ]


@app.callback(
        Output('adult-auto-table', 'style'),
        [Input('detect-target', 'value')])
def callback(detect):

    if detect == 'pupariation':
        return {'display': 'none'}

    elif detect == 'pupa-and-eclo':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    elif detect == 'death':
        return {
                'display': 'inline-block',
                'vertical-align': 'top',
                'margin': '10px',
                'width': '400px',
            }

    else:
        return {}


# ====================
#  Utility functions
# ====================
def seasoning(signals, signal_type, detect, size, sigma, smooth, weight,
        pupar_times, midpoints=None, weight_style=None):

    # Smooth the signals
    if smooth:
        signals = my_filter(signals, size=size, sigma=sigma)

    else:
        pass

    # Apply weight to the signals
    if weight:

        # Detection of pupariation
        if detect == 'pupariation' and signal_type == 'larva':

            # Step or ramp weight
            if weight_style == 'step':
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx, midpoint:] = 0

            elif weight_style == 'ramp':
                # Ramp filter
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx] = signals[well_idx] *  \
                            (-1 / midpoint * np.arange(len(signals.T)) + 1)
            else:
                pass

        # Not defined
        elif detect == 'pupariation' and signal_type == 'adult':
            pass

        # Detection of pupariation
        elif detect == 'pupa-and-eclo' and signal_type == 'larva':

            # Step or ramp weight
            if weight_style == 'step':
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx, midpoint:] = 0

            elif weight_style == 'ramp':
                # Ramp filter
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx] = signals[well_idx] *  \
                            (-1 / midpoint * np.arange(len(signals.T)) + 1)

            else:
                pass

        # Detection of eclosion
        elif detect == 'pupa-and-eclo' and signal_type == 'adult':
            """
            if pupar_times is not None:
                mask = -np.ones_like(signals, dtype=float)

                for i, event_timing in enumerate(pupar_times):
                    '''
                    # Step filter
                    mask[i, event_timing:] = 1
                    '''
                    # Ramp filter
                    lin_weight = np.linspace(
                            0, 1, len(signals.T) - event_timing)

                    mask[i, event_timing:] = lin_weight

                signals = signals * mask

            else:
                # Ramp filter
                signals = signals *  \
                        (np.arange(len(signals.T)) / len(signals.T))
            """

            # Step or ramp weight
            if weight_style == 'step':
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx, :midpoint] = 0

            elif weight_style == 'ramp':
                # Ramp filter
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx] = signals[well_idx] * (
                            1 / (len(signals.T) - midpoint)
                            * np.arange(len(signals.T))
                            - midpoint / (len(signals.T) - midpoint))
            else:
                pass

        # Not defined
        elif detect == 'death' and signal_type == 'larva':
            pass

        # Detection of death
        elif detect == 'death' and signal_type == 'adult':

            # Step or ramp weight
            if weight_style == 'step':
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx, midpoint:] = 0

            elif weight_style == 'ramp':
                # Ramp filter
                for well_idx, midpoint in enumerate(midpoints['midpoint']):
                    signals[well_idx] = signals[well_idx] *  \
                            (-1 / midpoint * np.arange(len(signals.T)) + 1)

            else:
                pass

    else:
        pass

    return signals


def detect_event(signals, thresholds, signal_type, detect):

    if detect == 'pupariation' and signal_type == 'larva':
        # Detect the falling of the signal
        # Scan the signal from the right hand side.
        auto_evals = (signals.shape[1]
                - (np.fliplr(signals) > thresholds).argmax(axis=1))
        # If the signal was not more than the threshold.
        auto_evals[auto_evals == signals.shape[1]] = 0

    elif detect == 'pupariation' and signal_type == 'adult':
        # Never evaluated
        pass

    elif detect == 'pupa-and-eclo' and signal_type == 'larva':
        # Detect the falling of the signal
        # Scan the signal from the right hand side.
        auto_evals = (signals.shape[1]
                - (np.fliplr(signals) > thresholds).argmax(axis=1))
        # If the signal was not more than the threshold.
        auto_evals[auto_evals == signals.shape[1]] = 0

    elif detect == 'pupa-and-eclo' and signal_type == 'adult':
        # Detect the rising of the signal
        # Compute event times from signals
        auto_evals = (signals > thresholds).argmax(axis=1)

    elif detect == 'death' and signal_type == 'larva':
        # Never evaluated
        pass

    elif detect == 'death' and signal_type == 'adult':
        # Scan the signal from the right hand side.
        auto_evals = (signals.shape[1]
                - (np.fliplr(signals) > thresholds).argmax(axis=1))

    return auto_evals


def load_blacklist(data_root, dataset_name, white=False):
    
    # Load a blacklist
    if os.path.exists(os.path.join(data_root, dataset_name, 'blacklist.csv')):
        blacklist = np.loadtxt(
                os.path.join(data_root, dataset_name, 'blacklist.csv'),
                dtype=np.int16, delimiter=',').flatten() == 1

        exist = True

    else:
        # Load a mask params
        with open(os.path.join(
                data_root, dataset_name, 'mask_params.json')) as f:
            params = json.load(f)

        blacklist = np.zeros(
                (params['n-rows']*params['n-plates'], params['n-clms'])
                ).flatten() == 1

        exist = False

    if white:
        return np.logical_not(blacklist), exist
    else:
        return blacklist, exist


def load_grouping_csv(data_root, dataset_name):

    if os.path.exists(os.path.join(data_root, dataset_name, 'grouping.csv')):
        groups = np.loadtxt(
                os.path.join(data_root, dataset_name, 'grouping.csv'),
                dtype=np.int16, delimiter=',').flatten()

        return [groups == i for i in range(1, groups.max() + 1)]

    else:
        return []


def day_and_night(timestamps):
    # Guard
    if timestamps is None:
        return []

    timestamps = pd.DataFrame(
            list(range(len(timestamps['Create time']))),
            index=timestamps['Create time'])
    timestamps.index = pd.to_datetime(timestamps.index)

    dates = timestamps.index.map(lambda t: t.date()).unique()

    shapes = []
    for date in dates:
        morning = pd.to_datetime(date.strftime('%x ') + '7:00')
        night = pd.to_datetime(date.strftime('%x ') + '19:00')

        if len(timestamps.loc[morning:night]) == 0:
            continue

        morning_idx = timestamps.loc[morning:night].iloc[0, 0]
        night_idx = timestamps.loc[morning:night].iloc[-1, 0]

        shapes.append(
            {
                'type': 'rect',
                'xref': 'x',
                'yref': 'paper',
                'x0': morning_idx,
                'y0': 0,
                'x1': night_idx,
                'y1': 1,
                'fillcolor': '#990000',
                'opacity': 0.1,
                'line': {'width': 0},
                'layer': 'below',
            }
        )

    return shapes


if __name__ == '__main__':
    app.run_server(debug=True)
