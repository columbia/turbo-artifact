import argparse
import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='file')
    args = parser.parse_args()
    app = dash.Dash()
    app.layout = html.Div(
        html.Div([
            html.Div(id='live-update-text'),
            dcc.Interval(
                id='interval-component',
                interval=1 * 1000,  # in milliseconds
                n_intervals=0
            )
        ])
    )


    @app.callback(Output('live-update-text', 'children'),
                  Input('interval-component', 'n_intervals'))
    def update(n):
        with open(args.file, 'rb') as fp:
            objs = pickle.load(fp)
            return html.Div(objs)


    app.run_server(debug=False, port="8080", host="127.0.0.1")
