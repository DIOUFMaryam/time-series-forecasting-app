#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:39:08 2024

@author: diouf.maryam5100@gmail.com

Start locally with `python myapp.py` or `shiny run myapp.py`.
On a hosting server, the server imports `app` from this file directly.
"""

from shiny import App, run_app
import ui
import server


app_ui = ui.create_ui()
app_server = server.create_server()

app = App(app_ui, app_server)


# Running the app
if __name__ == "__main__":
    # nest_asyncio lets the app run inside Spyder's console, which already
    # has an event loop. It is only needed when this file is run directly,
    # so a hosting server's own event loop is left untouched.
    import nest_asyncio
    nest_asyncio.apply()
    run_app(app)
