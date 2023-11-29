import os
from lxml import etree


def Parse(in_path):
    pathExtension = os.path.splitext(in_path)[1]
    traceControlFlow = []
    if pathExtension == ".MXML" or pathExtension == ".mxml":
        traceControlFlow = parseMXML(in_path)
    elif pathExtension == ".XES" or pathExtension == ".xes":
        doc = etree.parse(in_path)
        root = doc.getroot()
        if len(root.nsmap) < 1:
            traceControlFlow = parseXESNoXmlns(in_path)
        else:
            prefix = root.nsmap[None]
            traceControlFlow = parseXES(in_path, prefix)
    else:
        print("Error: the file must be .XES or .MXML! Please upload again.")
    return traceControlFlow


def parseMXML(in_path):
    traces = []
    events = set()
    tree = etree.parse(in_path)
    root = tree.getroot()
    process = root.xpath('./Process')[0]
    allTraces = process.xpath('./ProcessInstance')
    for case in allTraces:
        trace = []
        for subNode in case.iterfind('AuditTrailEntry'):
            dictEvent = parseMxmlEvent(subNode)
            trace.append(dictEvent)
            events.add(dictEvent["name"])
        traces.append(trace)
    traceControlFlow = onlyControlFlow(traces)
    return traceControlFlow


def parseXESNoXmlns(in_path):
    traces = []
    events = set()
    doc = etree.parse(in_path)
    allTraces = doc.xpath('trace')
    for case in allTraces:
        trace = []
        eventIntrace = case.xpath('./event')
        for subNode in eventIntrace:
            dictEvent = parseXesEventNoXmlns(subNode)
            trace.append(dictEvent)
            events.add(dictEvent["name"])
        traces.append(trace)
    traceControlFlow = onlyControlFlow(traces)
    return traceControlFlow


def parseXES(in_path, prefix):
    traces = [] # ALL trace
    events = set() # events
    doc = etree.parse(in_path)
    allTraces = doc.xpath('//pre:trace', namespaces={"pre": prefix})
    for case in allTraces:
        trace = []
        eventIntrace = case.xpath('./pre:event', namespaces={"pre": prefix})
        for subNode in eventIntrace:
            dictEvent = parseXesEvent(subNode, prefix)
            trace.append(dictEvent)
            events.add(dictEvent["name"])
        traces.append(trace)
    traceControlFlow = onlyControlFlow(traces)
    return traceControlFlow


def parseMxmlEvent(subNode):
    dictEvent = dict()
    for item in subNode:
        if item.tag == 'WorkflowModelElement':
            dictEvent['name'] = item.text.strip()
        elif item.tag == 'EventType':
            dictEvent['type'] = item.text.strip()
        elif item.tag == 'Timestamp' or item.tag == 'timestamp':
            dictEvent['timestamp'] = item.text.strip()
    return dictEvent


def parseXesEventNoXmlns(subNode):
    dictEvent = dict()
    stringNode = subNode.xpath('./string')
    dateNode = subNode.xpath('./date')

    for item in stringNode:
        if item.get('key') == 'concept:name':
            dictEvent['name'] = item.get('value').strip()
        elif item.get('key') == 'lifecycle:transition':
            dictEvent['type'] = item.get('value').strip()
    for item in dateNode:
        if item.get('key') == 'time:timestamp':
            dictEvent['timestamp'] = item.get('value').strip()
    return dictEvent


def parseXesEvent(subNode, prefix):
    dictEvent = dict()
    if prefix == None:
        stringNode = subNode.xpath('./pre:string')
        dateNode = subNode.xpath('./pre:date')
    else:
        stringNode = subNode.xpath('./pre:string', namespaces={"pre": prefix})
        dateNode = subNode.xpath('./pre:date', namespaces={"pre": prefix})
    for item in stringNode:
        if item.get('key') == 'concept:name':
            dictEvent['name'] = item.get('value').strip()
        elif item.get('key') == 'lifecycle:transition':
            dictEvent['type'] = item.get('value').strip()
    for item in dateNode:
        if item.get('key') == 'time:timestamp':
            dictEvent['timestamp'] = item.get('value').strip()
    return dictEvent


def onlyControlFlow(traces):
    listControl = []
    for t in traces:
        tempTrace = []
        for t_event in t:
            tempTrace.append(t_event['name'])
        listControl.append(tempTrace)
    return listControl