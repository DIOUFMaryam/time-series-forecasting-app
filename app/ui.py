#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:11:43 2024

@author: diouf.maryam5100@gmail.com

Migrated to the Shiny for Python 1.x API:
- ui.layout_sidebar(ui.panel_sidebar(...), ui.panel_main(...))
  -> ui.layout_sidebar(ui.sidebar(...), <main content>)
- ui.nav(...) -> ui.nav_panel(...)
- sidebar width is now a CSS size, not a 12-column grid width

Public demo: the company logo in the header is replaced by a short
note saying the bundled data is synthetic. Paths are relative to the
project folder, whatever folder the app is started from.
"""

from shiny import ui
import os


# Project root = the folder that contains app/, files/, kpi.ods...
# Paths are built from this file's location, not from the folder the
# app is started from, so the app also runs on a hosting server.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the directory where your Excel files are stored
DATA_DIR = os.path.join(ROOT_DIR, "files")


# Function to list all Excel files in the directory
def list_excel_files(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith('.xlsx'))


# Get the list of Excel files
excel_files = list_excel_files(DATA_DIR)


def create_ui():

    app_ui = ui.page_fluid(

        # CSS styling
        ui.tags.style(
            """

            .custom-header {
                background-color: black;
                color: #AEC6CF;
                padding: 10px;
                height: 80px
            }

            .header-title {
                text-align: left;
            }

            .header-note {
                text-align: right;
                font-size: 0.85rem;
                padding-top: 22px;
            }

            /* In Shiny 1.x the main area of layout_sidebar is <main class="main"> */
            .main {
                background-color: #f0f8ff;
                min-height: 1000px
            }

            .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
            visibility: visible;
            opacity: 1;
            }

            .nav-pills .nav-link {
                color: black;
            }
            .nav-pills .nav-link.active {
                background-color: black;
                color: white;
            }

            .progress {
                --bs-progress-height: 1.5rem;
            }
            .progress-bar {
                background-color: black;
                height: 28px
            }

            select#selected_kpis {
                height: 340px;
                border: 1px solid #ced4da;
            }

            .btn {
                background-color: black;
                color: white;
                border: none;
                padding: 10px 20px;
                text-align: center;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            .btn:hover {
                background-color: #ff6500;
                color: white;
            }
            .btn:active {
                background-color: #ff6500;
                color: white;
            }
            .btn:focus {
                outline: none;
                box-shadow: 0 0 10px #ff6500;
            }

            """
        ),

        # Header
        ui.div(
           ui.row(
               ui.column(6,
                         ui.div(
                             # window_title defaults to the title text
                             ui.panel_title("Time Series Forecasting"),
                             class_="header-title"
                         ),
               ),
               ui.column(6,
                         ui.div(
                             "Demo data: synthetic telecom KPIs",
                             class_="header-note"
                         ),
               ),
           ),
           class_="custom-header",
        ),

        ui.layout_sidebar(

            # Sidebar
            ui.sidebar(
                ui.input_select("file", "Choose Excel File", choices=excel_files),
                ui.div(
                    ui.input_select("column", "Select Column", choices=[]),
                    ui.output_ui("description"),
                ),
                ui.hr(),
                ui.navset_pill_list(
                    ui.nav_panel(
                        ui.HTML('<i class="fa fa-line-chart"></i> Time Series Plot'),
                        value="ts_plot_tab"
                    ),
                    ui.nav_panel(
                        ui.HTML('<i class="fa fa-table"></i> Data'),
                        value="data_tab"
                    ),
                    ui.nav_panel(
                        ui.HTML('<i class="fa fa-cogs"></i> Model'),
                        value="model_tab"
                    ),
                    ui.nav_panel(
                        ui.HTML('<i class="fa fa-calendar"></i> Forecast'),
                        value="previsions_tab"
                    ),
                    ui.nav_panel(
                        ui.HTML('<i class="fa fa-book"></i> Documentation'),
                        value="documentation_tab"
                    ),
                    id="sidebar_menu",
                    # Pills take the full sidebar width; panels are empty
                    # because the content is rendered in main_content.
                    well=False,
                    widths=(12, 12),
                ),
                bg="#ff6500",
                width=320,
            ),

            # Body (main area)
            ui.output_ui("main_content"),

        )
    )

    return app_ui
