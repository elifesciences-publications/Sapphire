# -*- coding: utf-8 -*-
# vim: set fileencoding=utf-8 :
# vim: set foldmethod=marker commentstring=\ \ #\ %s :
#
# Author:    Taishi Matsumura
# Created:   2018-08-19
#
# Copyright (C) 2018 Taishi Matsumura
#
import io
import time
import dash
import base64
import PIL.Image
import numpy as np
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


header = html.Header(html.H1('Mask Create Tool'))

uploader = dcc.Upload(
        id='uploader',
        children=html.Div(['Drag and Drop or ', html.A('Select A File')]),
        style={
            'width': '800px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'},
        multiple=False)

n_rows = dcc.Input(
        id='n-rows', type='number', value=8, max=100, min=0, size=5)
n_clms = dcc.Input(
        id='n-clms', type='number', value=12, max=100, min=0, size=5)
n_plates = dcc.Input(
        id='n-plates', type='number', value=3, max=10, min=0, size=5)
row_gap = dcc.Input(
        id='row-gap', type='number', value=1, max=10, min=0, size=5)
clm_gap = dcc.Input(
        id='clm-gap', type='number', value=1, max=10, min=0, size=5)
plate_gap = dcc.Input(
        id='plate-gap', type='number', value=71, max=100, min=0, size=5)
x = dcc.Input(
        id='x', type='number', value=0, max=1500, min=0, size=5)
y = dcc.Input(
        id='y', type='number', value=0, max=1500, min=0, size=5)
well_w = dcc.Input(
        id='well_w', type='number', value=0, max=1500, min=0, size=5)
well_h = dcc.Input(
        id='well_h', type='number', value=0, max=1500, min=0, size=5)

input_div = html.Div(
        id='input-div',
        children=[
            uploader, 'test string', n_rows, n_clms, n_plates,
            row_gap, clm_gap, plate_gap, x, y, well_w, well_h])

org_div = html.Div(
        [dcc.Graph(id='org-img', style={'visibility': 'hidden'})],
        id='org-div',
        style={
            'display': 'inline-block',
            'width': '33%',
        },
    )

mask_div = html.Div(
        id='mask-div',
        style={
            'display': 'inline-block',
            'width': '66%',
        },
    )

images_div = html.Div(
        id='images-div',
        children=[org_div, mask_div],
        style={'width': '1200px'},  # for 1280x1024 display
    )

app = dash.Dash()
app.layout = html.Div([header, input_div, images_div])
app.css.append_css(
        {'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})


@app.callback(
        Output('org-div', 'children'),
        [Input('uploader', 'contents')])
def update_images_div(data_uri):
    if data_uri is None:
        return
    imghash = data_uri.split(',')[1]
    img = PIL.Image.open(io.BytesIO(base64.b64decode(imghash)))
    height = np.array(img).shape[0]
    width = np.array(img).shape[1]
    graph = dcc.Graph(
        id='org-img',
        figure = {
            'data': [go.Scatter(x=[0], y=[0], mode='lines+markers')],
            'layout': {
                'width': 400,
                'height': 700,
                'margin': go.layout.Margin(l=40, b=40, t=26, r=10),
                'xaxis': {
                    'range': (0, width),
                    'scaleanchor': 'y',
                    'scaleratio': 1,
                },
                'yaxis': {
                    'range': (0, height),
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
                    'source': data_uri,
                }],
                'dragmode': 'select',
            }
        },
        style={
            'display': 'inline-block',
            'width': '100%',
            'visibility': 'visible',
        }
    )
    return [graph]


@app.callback(
        Output('x', 'value'),
        [Input('org-img', 'selectedData')])
def update_x(selected_data):
    if selected_data is None:
        return
    range_x = np.array(selected_data['range']['x']).astype(int)
    return range_x[0]


@app.callback(
        Output('y', 'value'),
        [Input('org-img', 'selectedData')])
def update_y(selected_data):
    if selected_data is None:
        return
    range_y = np.array(selected_data['range']['y']).astype(int)
    return range_y[0]



@app.callback(
        Output('well_w', 'value'),
        [Input('org-img', 'selectedData')])
def update_well_w(selected_data):
    if selected_data is None:
        return
    range_x = np.array(selected_data['range']['x']).astype(int)
    return range_x[1] - range_x[0]


@app.callback(
        Output('well_h', 'value'),
        [Input('org-img', 'selectedData')])
def update_well_h(selected_data):
    if selected_data is None:
        return
    range_y = np.array(selected_data['range']['y']).astype(int)
    return range_y[1] - range_y[0]


@app.callback(
        Output('mask-div', 'children'),
        [Input('n-rows', 'value'),
            Input('n-clms', 'value'),
            Input('n-plates', 'value'),
            Input('row-gap', 'value'),
            Input('clm-gap', 'value'),
            Input('plate-gap', 'value'),
            Input('x', 'value'),
            Input('y', 'value'),
            Input('well_w', 'value'),
            Input('well_h', 'value')],
        [State('org-img', 'figure')])
def draw_images(
        n_rows, n_clms, n_plates,
        gap_r, gap_c, gap_p, x, y, well_w, well_h, figure):
    if figure is None:
        return
    # Get base64ed hash of original image
    orgimg_uri = figure['layout']['images'][0]['source']
    imghash = orgimg_uri.split(',')[1]

    # Transform hash to ndarray
    org_img = np.array(PIL.Image.open(io.BytesIO(base64.b64decode(imghash))))

    # Parameters
    height, width = org_img.shape[0], org_img.shape[1]
    count = 0

    # Mask create loop
    mask = -1 * np.ones_like(org_img)
    for n in range(n_plates):
        for idx_r in range(n_rows):
            for idx_c in range(n_clms):
                c1 = x + idx_c*(well_w + gap_c)
                c2 = c1 + well_w
                r1 = y + idx_r*(well_h + gap_r) + n*(n_rows*well_h + gap_p) + gap_r*(n - 1)
                r2 = r1 + well_h
                mask[r1:r2, c1:c2] = count
                count += 1

    grayed = PIL.Image.fromarray(
            np.flipud(np.where(mask>=0, 255, 0).astype(np.uint8)))
    masked = PIL.Image.fromarray(
            np.flipud(np.where(mask>=0, 1, 0).astype(np.uint8)) * org_img)
    mask_buf = io.BytesIO()
    masked_buf = io.BytesIO()
    grayed.save(mask_buf, format='PNG')
    masked.save(masked_buf, format='PNG')

    np.save('static/mask.npy', mask.astype(np.int16))

    mask_img = dcc.Graph(
        id='mask-img',
        figure={
            'data': [go.Scatter(x=[0], y=[0], mode='lines+markers')],
            'layout': {
                'width': 400,
                'height': 700,
                'margin': go.layout.Margin(l=40, b=40, t=26, r=10),
                'xaxis': {
                    'range': (0, width),
                    'scaleanchor': 'y',
                    'scaleratio': 1,
                },
                'yaxis': {
                    'range': (0, height),
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
                    'source': 'data:image/png;base64,{}'.format(
                        base64.b64encode(mask_buf.getvalue()).decode('utf-8')),
                }],
                'dragmode': 'select',
            }
        },
        style={
            'display': 'inline-block',
            'width': '50%',
        }
    )

    masked_img = dcc.Graph(
        id='masked-img',
        figure={
            'data': [go.Scatter(x=[0], y=[0], mode='lines+markers')],
            'layout': {
                'width': 400,
                'height': 700,
                'margin': go.layout.Margin(l=40, b=40, t=26, r=10),
                'xaxis': {
                    'range': (0, width),
                    'scaleanchor': 'y',
                    'scaleratio': 1,
                },
                'yaxis': {
                    'range': (0, height),
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
                    'source': 'data:image/png;base64,{}'.format(
                        base64.b64encode(masked_buf.getvalue()).decode('utf-8')),
                }],
                'dragmode': 'select',
            }
        },
        style={
            'display': 'inline-block',
            'width': '50%',
        }
    )

    return [masked_img, mask_img]

if __name__ == '__main__':
    app.run_server(debug=False)
