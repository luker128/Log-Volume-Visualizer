# Log-Volume-Visualizer
Log file exploration tool. Plots volume of logs lines in time.

## Prerequisites

Required runtime dependencies are:
 - Python3 interpreter
 - tkinter library

On debian based systems those can be installed by:

```bash
sudo apt install python3 python3-tk
```

## Usage

For a basic log line count analysis just provide log file in command line:

```bash
./logvol.py <logfile>
```

To plot number of ocurrences of speciffic keyword use:

```bash
./logvol.py <logfile> <keyword>
```

## User interface

Use mouse wheel to zoom the plot.

Dragging with right mouse button scrolls the plot.

Left mouse button selects a portion of plot.

Clicking right mouse button without dragging will show context menu

Selected portion of log file can be viewed in external editor. This will produce a temporary file
containing log lines from selected time frame. EDITOR environment variable will be used to execute
prefered text editor. Application will be suspended until the external editor is closed. Temporary
file will be deleted when closing editor.
