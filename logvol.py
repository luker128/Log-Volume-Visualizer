#! /bin/env python3
import tkinter as tk
from tkinter import ttk
import sys
import time
import datetime
import re
import subprocess
import os
import tempfile
import argparse

####
# Log analysis part
#####

class TimestampFormat:

    def __init__(self, start, end, timeFormat):
        self.start = start
        self.end = end
        self.timeFormat = timeFormat

    def extractTimestamp(self, line):
        return line[self.start:self.end]

    def parseTimestamp(self, timestamp):
        try:
            return datetime.datetime.strptime(timestamp, self.timeFormat)
        except ValueError:
            return None

    def buildTimestamp(self, timeValue):
        return timeValue.strftime(self.timeFormat)

    def detect(line, override):
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%m-%d %H:%M:%S"
        ]
        if override is not None:
            formats = [override]
        for timeFormat in formats:
            regex = datetime.datetime(1111,11,11,11,11,11,111111).strftime(timeFormat).replace("1", r"\d")
            m = re.search(regex, line)
            if m is not None:
                tf = TimestampFormat(m.start(), m.end(), timeFormat)
                return tf
        return None

class LogAnalyzer:

    def __init__(self):
        self.dataSeriesLabels = ["Volume"]
        self.dataSeriesDefinitions = [lambda l : 1]

    def addKeyword(self, keyword):
        self.dataSeriesLabels.append(keyword)
        self.dataSeriesDefinitions.append(lambda l, k=keyword: 1 if k in l else 0)

    def addRegex(self, regex):
        self.dataSeriesLabels.append(regex)
        regexObj = re.compile(regex)
        self.dataSeriesDefinitions.append(lambda l, r=regexObj: 1 if r.search(l) is not None else 0)

    def buildDataSeriesInfo(self):
        return self.dataSeriesLabels

    def appendData(self, data, count):
        for dataSeries, countSeries in zip(data, count):
            dataSeries.append(countSeries)

    def analyzeLog(self, filename, timestampFormatOverride):
        data = [[]] * len(self.dataSeriesLabels)
        labels = []
        filePositions = []
        count = [0] * len(self.dataSeriesLabels)
        lastTimestamp = ""
        lastTimeValue = 0
        lines = 0
        timestampFormat = None
        timeBefore = time.time()
        startFilePosition = 0
        filePositionBeforeReadLine = 0
        with open(filename) as file:
            while True:
                line = file.readline()
                if line == "":
                    break
                lines += 1
                if timestampFormat is None:
                    timestampFormat = TimestampFormat.detect(line, timestampFormatOverride)
                    if timestampFormat is None:
                        print("Unrecognized timestamp format")
                        print("Please rerun providing timestamp format with '-f'")
                        return None
                    lastTimestamp = timestampFormat.extractTimestamp(line)
                    lastTimeValue = timestampFormat.parseTimestamp(lastTimestamp)
                    print("Detected timestamp '{}' with format '{}'".format(line[timestampFormat.start:timestampFormat.end], timestampFormat.timeFormat))
                timestamp = line[timestampFormat.start:timestampFormat.end]
                if timestamp != lastTimestamp:
                    timeValue = timestampFormat.parseTimestamp(timestamp)
                    if timeValue is not None:
                        self.appendData(data, count)
                        labels.append(lastTimeValue)
                        filePositions.append( (filename, lines, startFilePosition, filePositionBeforeReadLine) )
                        count = [0] * len(self.dataSeriesLabels)
                        timeDelta = (timeValue - lastTimeValue).total_seconds()
                        if timeDelta > 1.5:
                            samplesToInsert = int(timeDelta) - 1
                            for i in range(samplesToInsert):
                                self.appendData(data, [0] * len(self.dataSeriesLabels))
                                intermediateTimeValue = lastTimeValue + datetime.timedelta(seconds = i+1)
                                labels.append(intermediateTimeValue)
                                filePositions.append( (filename, lines, startFilePosition, filePositionBeforeReadLine) )
                        lastTimeValue = timeValue
                        lastTimestamp = timestamp
                        startFilePosition = filePositionBeforeReadLine
                    else:
                        print("timestamp not found in line", line)
                        # timestamp not found, skipping this line
                        pass
                for i, countGetter in enumerate(self.dataSeriesDefinitions):
                    count[i] += countGetter(line)
                filePositionBeforeReadLine += len(line)
        self.appendData(data, count)
        labels.append(lastTimeValue)
        filePositions.append( (filename, lines, startFilePosition, filePositionBeforeReadLine) )
        timeAfter = time.time()
        timeElapsed = timeAfter - timeBefore
        print("{:,} lines processed into {:,} samples in {:.2f} seconds".format(lines, len(labels), timeElapsed))
        return data, labels, filePositions, self.buildDataSeriesInfo()

def exportDataToFile(data, filename):
    with open(filename, "w") as file:
        dataSeries, labels, filePositions, seriesInfo = data
        file.write("file positions;timestamp")
        for info in seriesInfo:
            file.write(";{}".format(info))
        file.write("\n")
        for i in range(len(labels)):
            file.write("{}:{}:{}:{};".format(*filePositions[i]))
            file.write(str(labels[i]))
            for series in dataSeries:
                file.write(";{}".format(series[i]))
            file.write("\n")


def importDataFromFile(filename):
    with open(filename, "r") as file:
        allLines = file.readlines()
        header = allLines[0].strip()
        headerFields = header.split(";")
        seriesInfo = headerFields[2:]
        lines = file.readlines()
        dataSeries = [[]] * len(seriesInfo)
        labels = []
        filePositions = []
        for line in allLines[1:]:
            line = line.strip()
            fields = line.split(";")
            fpFields = fields[0].split(":")
            fp = [fpFields[0]] + [int(f) for f in fpFields[1:]]
            filePositions.append(fp)
            labels.append(datetime.datetime.strptime(fields[1], "%Y-%m-%d %H:%M:%S"))
            for i in range(len(fields[2:])):
                dataSeries[i].append(int(fields[2+i]))
        return dataSeries, labels, filePositions, seriesInfo

####
# Plot drawing part
#####

class Plot:

    def __init__(self, data):
        self.dataSeries, self.labels, self.filePositions, self.dataSeriesInfo = data
        self.currentDataSeries = 0
        self.samples = self.dataSeries[self.currentDataSeries]
        self.margin = 8
        self.visibleRange = (0, len(self.samples))
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.mouse_dragging = None
        self.selection = None

    def getCurrentDataSeries(self):
        return self.currentDataSeries

    def getDataSeriesInfo(self):
        return self.dataSeriesInfo

    def selectDataSeries(self, series):
        self.currentDataSeries = series
        self.samples = self.dataSeries[self.currentDataSeries]
        self.draw()

    def exportSelection(self):
        # This function doesn't really belong in Plot class, but all input data is available here,
        # so it was most convinient to put it here for now.
        if self.selection is None:
            return
        startOffset = self.filePositions[self.selection[0]][2]
        endOffset = self.filePositions[self.selection[1]][3]
        inFile = self.filePositions[self.selection[0]][0]
        outFileFd, outFile = tempfile.mkstemp()
        print("Exporting offsets {} - {} from {} ...".format(startOffset, endOffset, inFile))
        with open(inFile, "r") as file_in:
            with os.fdopen(outFileFd, "w") as file_out:
                file_in.seek(startOffset)
                chunk = file_in.read(endOffset - startOffset)
                file_out.write(chunk)
        print("Starting editor")
        subprocess.run(["$EDITOR {}".format(outFile)], shell = True)
        os.remove(outFile)

    def setCanvas(self, canvas):
        self.canvas = canvas
        self.setSize(int(canvas.__getitem__('width')), int(canvas.__getitem__('height')))

    def hasSelection(self):
        return self.selection is not None

    def clearSelection(self):
        self.selection = None
        self.draw()

    def getStatistics(self):
        sampleAtCursor = self.getSampleAt(self.last_mouse_x)
        totalStart = self.labels[0]
        totalEnd = self.labels[-1]
        totalDuration = self.labels[-1] - self.labels[0]
        totalTotal = sum(self.samples)
        totalMax = max(self.samples)
        totalMin = min(self.samples)
        totalAvg = round(totalTotal / len(self.samples), 3)
        visibleStart = self.labels[self.visibleRange[0]]
        visibleEnd = self.labels[self.visibleRange[1]-1]
        visibleDuration = self.labels[self.visibleRange[1]-1] - self.labels[self.visibleRange[0]]
        visibleTotal = sum(self.samples[self.visibleRange[0]:self.visibleRange[1]])
        visibleMax = max(self.samples[self.visibleRange[0]:self.visibleRange[1]])
        visibleMin = min(self.samples[self.visibleRange[0]:self.visibleRange[1]])
        visibleAvg = round(visibleTotal / (self.visibleRange[1] - self.visibleRange[0]), 3)
        if self.selection is not None:
            selectionStart = self.labels[self.selection[0]]
            selectionEnd = self.labels[self.selection[1]]
            selectionDuration =  self.labels[self.selection[1]] - self.labels[self.selection[0]]
            selectionTotal = sum(self.samples[self.selection[0]:self.selection[1]])
            selectionMax = max(self.samples[self.selection[0]:self.selection[1]])
            selectionMin = min(self.samples[self.selection[0]:self.selection[1]])
            selectionAvg = round(selectionTotal / (self.selection[1] - self.selection[0]), 3)
        return {"cursor-value":       sampleAtCursor[1],
                "cursor-label":       sampleAtCursor[2].strftime("%H:%M:%S"),
                "total-start":        totalStart.strftime("%H:%M:%S"),
                "total-end":          totalEnd.strftime("%H:%M:%S"),
                "total-duration":     totalDuration,
                "total-total":        "{:,}".format(totalTotal),
                "total-min":          "{:,}".format(totalMin),
                "total-max":          "{:,}".format(totalMax),
                "total-avg":          "{:,}".format(totalAvg),
                "visible-start":      visibleStart.strftime("%H:%M:%S"),
                "visible-end":        visibleEnd.strftime("%H:%M:%S"),
                "visible-duration":   visibleDuration,
                "visible-total":      "{:,}".format(visibleTotal),
                "visible-min":        "{:,}".format(visibleMin),
                "visible-max":        "{:,}".format(visibleMax),
                "visible-avg":        "{:,}".format(visibleAvg),
                "selection-start":    selectionStart.strftime("%H:%M:%S") if self.hasSelection() else "-",
                "selection-end":      selectionEnd.strftime("%H:%M:%S") if self.hasSelection() else "-",
                "selection-duration": selectionDuration if self.hasSelection() else "-",
                "selection-total":    "{:,}".format(selectionTotal) if self.hasSelection() else "-",
                "selection-min":      "{:,}".format(selectionMin) if self.hasSelection() else "-",
                "selection-max":      "{:,}".format(selectionMax) if self.hasSelection() else "-",
                "selection-avg":      "{:,}".format(selectionAvg) if self.hasSelection() else "-",
        }

    def mouse_motion(self, x, y):
        self.last_mouse_x = x
        self.last_mouse_y = y

    def mouse_wheel_down(self):
        self.zoom(self.last_mouse_x, 0.75)

    def mouse_wheel_up(self):
        self.zoom(self.last_mouse_x, 1.25)

    def mouse_drag(self, x, y, button):
        self.last_mouse_x = x
        self.last_mouse_y = y
        if button == 2:
            delta = self.mouse_drag_origin - x
            self.mouse_drag_origin = x
            self.scroll(delta)
        elif button == 1:
            start = self.getIndexAt(self.mouse_drag_origin)
            end = self.getIndexAt(x)
            if start != end:
                self.selection = (min(start, end), max(start, end))
            else:
                self.selection = None
            self.draw()

    def mouse_press(self, x, y, button, pressed):
        if pressed:
            self.mouse_drag_origin = x
            self.mouse_dragging = button
        else:
            self.mouse_dragging = None

    def scroll(self, amount):
        numVisibleSamples = self.visibleRange[1] - self.visibleRange[0]
        pixelsPerSample = self.width / float(numVisibleSamples)
        amount /= pixelsPerSample
        self.visibleRange = (int(self.visibleRange[0] + amount), int(self.visibleRange[1] + amount))
        if self.visibleRange[0] < 0:
            self.visibleRange = (0, numVisibleSamples)
        if self.visibleRange[1] > len(self.samples):
            self.visibleRange = (len(self.samples) - numVisibleSamples, len(self.samples))
        self.draw()

    def zoom(self, pivot, scale):
        pivotIndex = self.getIndexAt(pivot)
        currentRange = self.visibleRange[1] - self.visibleRange[0]
        pivotPosition = (pivotIndex - self.visibleRange[0]) / currentRange
        newRange = int(currentRange * scale)
        newStart = int(max(0, pivotIndex - newRange * pivotPosition))
        newEnd = int(newStart + newRange)
        if newEnd > len(self.samples):
            newEnd = len(self.samples)
            newStart = newEnd - newRange
        self.visibleRange = (newStart, newEnd)
        self.visibleRange = (max(0, self.visibleRange[0]), min(self.visibleRange[1], len(self.samples)))
        self.visibleRange = (self.visibleRange[0], max(self.visibleRange[0]+10, self.visibleRange[1]))
        self.draw()

    def zoomOut(self):
        self.visibleRange = (0, len(self.samples))
        self.draw()

    def zoomIntoSelection(self):
        if self.selection is not None:
            self.visibleRange = self.selection
            self.draw()

    def setSize(self, w, h):
        self.width = w
        self.height = h
        self.barWidth = (self.width / len(self.samples))

    def aggregateSamples(self, samples):
        return max(samples)

    def getIndexAt(self, x):
        x -= self.margin
        numVisibleSamples = self.visibleRange[1] - self.visibleRange[0]
        pixelsPerSample = self.width / float(numVisibleSamples)
        index = int(x / pixelsPerSample) + self.visibleRange[0]
        return index

    def getSampleAt(self, x):
        index = self.getIndexAt(x)
        nextIndex = self.getIndexAt(x+1)
        if nextIndex == index:
            return index, self.samples[index], self.labels[index]
        return index, self.aggregateSamples(self.samples[index:nextIndex]), self.labels[index]

    def draw(self):
        c = self.canvas
        c.delete("all")
        w, h = self.width, self.height
        bar_w = self.barWidth
        x, y = 0, 0
        c.create_rectangle((x, y), (x+w, y+h), fill='silver')

        x += self.margin
        y += self.margin
        w -= self.margin * 2
        h -= self.margin * 2 + 80
        c.create_rectangle((x, y), (x+w, y+h), fill='white')
        # exclude border
        x += 1
        y += 1
        w -= 2
        h -= 2

        # For scaling plot verticaly we take maximum value of all samples not just visible part.
        # This could be changed in future to implement auto scaling
        max_sample = max(self.samples)
        if max_sample != 0:
            y_scale = h / max_sample
        else:
            y_scale = 100
        lastLabelAt = 0
        start_x = x

        max_grid = 16
        grid_space = w / max_grid
        index_a, _, _ = self.getSampleAt(x)
        index_b, _, _ = self.getSampleAt(x+grid_space)
        grid_space_index = index_b - index_a
        if grid_space_index == 0:
            grid_space_index = 1
        last_grid_at_index = ((index_a // grid_space_index)) * grid_space_index
        x = start_x
        for bar_x in range(w+1):
            index, sample, label = self.getSampleAt(x)
            if last_grid_at_index + grid_space_index < index:
                verticalGridLine = True
                last_grid_at_index = index
                labelText = label.strftime("%H:%M:%S")
                c.create_text(x + bar_w/2, y + h+8, text=labelText, font=("Arial", 9), fill="blue", angle=45, anchor='e')
            else:
                verticalGridLine = False
            if self.selection is not None and index > self.selection[0] and index < self.selection[1]:
                if verticalGridLine:
                    c.create_line((x, y), (x, y+h), fill='#ddd')
                else:
                    c.create_line((x, y), (x, y+(h-sample*y_scale)), fill='yellow')
                c.create_line((x, y+h+1), (x, y+(h-sample*y_scale)), fill='orange')
            elif sample != 0:
                if verticalGridLine:
                    c.create_line((x, y), (x, y+h), fill='#eee')
                c.create_line((x, y+h+1), (x, y+(h-sample*y_scale)), fill='green')
            x += 1

class Gui:

    def addStatsField(self, statKey, label, col, row, section):
        if section in self.statLabelSections:
            frame = self.statLabelSections[section]
        else:
            frame = ttk.LabelFrame(self.frm, padding=10, text=section)
            frame.grid(column=len(self.statLabelSections), row=0, padx=10)
            self.statLabelSections[section] = frame
        labelLabel = ttk.Label(frame, text=label)
        labelLabel.grid(column=col*2, row=row, sticky=tk.EW, padx=5, pady=2)
        valueLabel = ttk.Label(frame, text="-")
        valueLabel.grid(column=col*2+1, row=row, sticky=tk.EW, padx=10, pady=2)
        self.statLabels[statKey] = valueLabel

    def __init__(self, plot):
        self.plot = plot
        self.root = tk.Tk()
        self.root.geometry('1280x1024')
        self.root.title('Log Volume Visualizer')
        self.canvas = tk.Canvas(self.root, width=1280, height=720, bg='white')
        self.canvas.pack(anchor=tk.CENTER, fill="both", expand=True)
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.pack(fill="both", expand=True)

        self.statLabels = {}
        self.statLabelSections = {}
        self.addStatsField("cursor-value",       "Cursor Value:",       0, 0, "Cursor")
        self.addStatsField("cursor-label",       "Cursor Timestamp:",   0, 1, "Cursor")

        self.addStatsField("total-start",        "Log start:",          1, 0, "Log")
        self.addStatsField("total-end",          "Log end:",            1, 1, "Log")
        self.addStatsField("total-duration",     "Log duration:",       1, 2, "Log")
        self.addStatsField("total-total",        "Log total:",          1, 3, "Log")
        self.addStatsField("total-min",          "Log minumum:",        1, 4, "Log")
        self.addStatsField("total-max",          "Log maximum:",        1, 5, "Log")
        self.addStatsField("total-avg",          "Log average:",        1, 6, "Log")

        self.addStatsField("visible-start",      "Window start:",       2, 0, "Window")
        self.addStatsField("visible-end",        "Window end:",         2, 1, "Window")
        self.addStatsField("visible-duration",   "Window duration:",    2, 2, "Window")
        self.addStatsField("visible-total",      "Window total:",       2, 3, "Window")
        self.addStatsField("visible-min",        "Window minumum:",     2, 4, "Window")
        self.addStatsField("visible-max",        "Window maximum:",     2, 5, "Window")
        self.addStatsField("visible-avg",        "Window average:",     2, 6, "Window")

        self.addStatsField("selection-start",    "Selection start:",    3, 0, "Selection")
        self.addStatsField("selection-end",      "Selection end:",      3, 1, "Selection")
        self.addStatsField("selection-duration", "Selection duration:", 3, 2, "Selection")
        self.addStatsField("selection-total",    "Selection total:",    3, 3, "Selection")
        self.addStatsField("selection-min",      "Selection minumum:",  3, 4, "Selection")
        self.addStatsField("selection-max",      "Selection maximum:",  3, 5, "Selection")
        self.addStatsField("selection-avg",      "Selection average:",  3, 6, "Selection")

        self.selection_context_menu = tk.Menu(self.root, tearoff=0)
        self.selection_context_menu.add_command(label="Zoom into", command=self.menu_action_zoom_into)
        self.selection_context_menu.add_command(label="Zoom out", command=self.menu_action_zoom_out)
        self.selection_context_menu.add_command(label="View in editor", command=self.menu_action_export)
        self.selection_context_menu.add_command(label="Unselect", command=self.menu_action_unselect)
        self.generic_context_menu = tk.Menu(self.root, tearoff=0)
        self.generic_context_menu.add_command(label="Zoom out", command=self.menu_action_zoom_out)

        for i, dataSeries in enumerate(self.plot.getDataSeriesInfo()):
            self.generic_context_menu.add_command(label="Switch to data series '{}'".format(dataSeries),
                                                  command=lambda selectedDs=i: self.menu_action_selectSeries(selectedDs))
            self.selection_context_menu.add_command(label="Switch to data series '{}'".format(dataSeries),
                                                  command=lambda selectedDs=i: self.menu_action_selectSeries(selectedDs))
        self.plot.setCanvas(self.canvas)
        self.canvas.bind('<Configure>', self.window_resize)
        self.canvas.bind("<Motion>", self.motion)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Button-4>", self.mouse_wheel)
        self.canvas.bind("<Button-5>", self.mouse_wheel)
        self.canvas.bind("<B1-Motion>", self.mouse_drag_left)
        self.canvas.bind("<ButtonPress-1>", self.mouse_button_left)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_button_release_left)
        self.canvas.bind("<B3-Motion>", self.mouse_drag_right)
        self.canvas.bind("<ButtonPress-3>", self.mouse_button_right)
        self.canvas.bind("<ButtonRelease-3>", self.mouse_button_release_right)
        self.startedRightMouseDrag = False

    def mainloop(self):
        self.root.mainloop()

    def setPlot(self, plot):
        self.plot = plot

    def window_resize(self, event):
        self.plot.setSize(event.width, event.height)
        self.plot.draw()
        self.updateStatsPanel()

    def motion(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.plot.mouse_motion(x, y)
        self.updateStatsPanel()

    def updateStatsPanel(self):
        stats = self.plot.getStatistics()
        for k, v in stats.items():
            if k in self.statLabels:
                self.statLabels[k]["text"] = v

    def mouse_wheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.plot.mouse_wheel_up()
            self.updateStatsPanel()
        if event.num == 4 or event.delta == 120:
            self.plot.mouse_wheel_down()
            self.updateStatsPanel()

    def mouse_drag_left(self, event):
        self.plot.mouse_drag(event.x, event.y, 1)
        self.updateStatsPanel()

    def mouse_button_left(self, event):
        self.plot.mouse_press(event.x, event.y, 1, True)

    def mouse_button_release_left(self, event):
        self.plot.mouse_press(event.x, event.y, 1, False)

    def mouse_drag_right(self, event):
        self.startedRightMouseDrag = True
        self.plot.mouse_drag(event.x, event.y, 2)
        self.updateStatsPanel()

    def mouse_button_right(self, event):
        self.plot.mouse_press(event.x, event.y, 2, True)
        self.startedRightMouseDrag = False

    def mouse_button_release_right(self, event):
        if not self.startedRightMouseDrag:
            self.show_context_menu(event)
        self.plot.mouse_press(event.x, event.y, 2, False)
        self.startedRightMouseDrag = False

    def show_context_menu(self, event):
        if self.plot.hasSelection():
            self.selection_context_menu.post(event.x_root, event.y_root)
        else:
            self.generic_context_menu.post(event.x_root, event.y_root)

    def menu_action_zoom_into(self):
        self.plot.zoomIntoSelection()
        self.updateStatsPanel()

    def menu_action_zoom_out(self):
        self.plot.zoomOut()
        self.updateStatsPanel()

    def menu_action_export(self):
        self.plot.exportSelection()

    def menu_action_unselect(self):
        self.plot.clearSelection()
        self.updateStatsPanel()

    def menu_action_selectSeries(self, dataSeries):
        self.plot.selectDataSeries(dataSeries)
        self.updateStatsPanel()

def main():
    parser = argparse.ArgumentParser(description="Draw a graph of amount of logs in time")
    parser.add_argument("filename", default=None, nargs='?')
    parser.add_argument("-f", "--timestamp-format", help="see 'man date' for format description")
    parser.add_argument("-k", "--keyword", action="append", default=[])
    parser.add_argument("-e", "--regex", action="append", default=[])
    parser.add_argument("-o", "--data-output")
    parser.add_argument("-i", "--data-input")
    parser.add_argument("-q", "--no-gui", action="store_true")
    args = parser.parse_args()
    if args.data_input is None:
        # No data input file provided - run analysis
        analyzer = LogAnalyzer()
        for keyword in args.keyword:
            analyzer.addKeyword(keyword)
        for regex in args.regex:
            analyzer.addRegex(regex)
        data = analyzer.analyzeLog(args.filename, args.timestamp_format)
        if data is None:
            return
    else:
        data = importDataFromFile(args.data_input)
    if args.data_output is not None:
        exportDataToFile(data, args.data_output)
    if not args.no_gui:
        plot = Plot(data)
        gui = Gui(plot)
        gui.mainloop()

if __name__ == "__main__":
    main()
