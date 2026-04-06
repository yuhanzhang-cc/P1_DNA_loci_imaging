repoRoot = "/Users/mac/Developer/P1_DNA_loci_imaging";
rawTifDir = repoRoot + "/data/raw/tif/";
rawPngDir = repoRoot + "/data/raw/png/";
trackmateXmlDir = repoRoot + "/data/raw/logs/trackmate_xml/";
stateDir = repoRoot + "/data/raw/logs/annotation_state/";

macro "Prepare Annotation" {
    if (nImages == 0) exit("No image is open.");

    File.makeDirectory(rawTifDir);
    File.makeDirectory(rawPngDir);
    File.makeDirectory(trackmateXmlDir);
    File.makeDirectory(stateDir);

    originalTitle = getTitle();
    getDimensions(width, height, channels, slices, frames);
    Stack.getPosition(currentChannel, currentSlice, currentFrame);
    getPixelSize(unit, pixelWidth, pixelHeight, voxelDepth);

    sourcePath = File.openDialog("Select the original source file (.tif/.tiff/.nd2/.mvd2) used for this image");
    if (sourcePath == "") exit("Cancelled.");

    defaultSourceName = File.getName(sourcePath);
    if (endsWith(sourcePath, ".mvd2")) defaultSourceName = originalTitle;

    Dialog.create("Annotation Metadata");
    Dialog.addString("source_file_name", defaultSourceName, 60);
    Dialog.addNumber("channel_index", currentChannel);
    Dialog.addCheckbox("apply_max_intensity_z_projection", slices > 1);
    Dialog.show();

    sourceFileName = Dialog.getString();
    selectedChannel = Dialog.getNumber();
    doMaxProjection = Dialog.getCheckbox();

    if (endsWith(sourcePath, ".mvd2") && doMaxProjection && !startsWith(sourceFileName, "MAX_"))
        sourceFileName = "MAX_" + sourceFileName;

    sampleNumber = nextSampleNumber(rawTifDir);
    sampleBase = "sample_" + sampleNumber;
    tifPath = rawTifDir + sampleBase + ".tif";
    pngPath = rawPngDir + sampleBase + ".png";
    xmlPath = trackmateXmlDir + sampleBase + ".xml";
    metaPath = stateDir + sampleBase + ".meta.txt";

    run(
        "Duplicate...",
        "title=" + sampleBase + "_working duplicate channels=" + selectedChannel + "-" + selectedChannel +
        " slices=1-" + slices + " frames=" + currentFrame + "-" + currentFrame
    );
    selectWindow(sampleBase + "_working");

    if (doMaxProjection && slices > 1) {
        run("Z Project...", "projection=[Max Intensity]");
        close(sampleBase + "_working");
        selectWindow("MAX_" + sampleBase + "_working");
        rename(sampleBase + "_prepared");
        zAxisValue = 0;
    } else {
        rename(sampleBase + "_prepared");
        if (slices > 1)
            zAxisValue = currentSlice;
        else
            zAxisValue = 0;
    }

    run("Properties...", "channels=1 slices=1 frames=1");
    saveAs("Tiff", tifPath);

    showMessage(
        "TrackMate Save Path",
        "The processed TIFF has been saved to:\n" + tifPath +
        "\n\nTrackMate XML should be saved to:\n" + xmlPath +
        "\n\nAfter closing TrackMate, the macro will save the annotated PNG and write the metadata sidecar."
    );

    run("TrackMate");

    selectWindow(sampleBase + "_prepared");
    saveAs("PNG", pngPath);

    currentDate = isoDate();
    sidecar = "";
    sidecar += "date=" + currentDate + "\n";
    sidecar += "source_path=" + sourcePath + "\n";
    sidecar += "source_file_name=" + sourceFileName + "\n";
    sidecar += "frame=" + currentFrame + "\n";
    sidecar += "z_axis=" + zAxisValue + "\n";
    sidecar += "channel=" + selectedChannel + "\n";
    sidecar += "raw_tif_name=" + sampleBase + ".tif\n";
    sidecar += "raw_csv_name=" + sampleBase + ".csv\n";
    sidecar += "raw_png_name=" + sampleBase + ".png\n";
    sidecar += "x_pixel=" + width + "\n";
    sidecar += "y_pixel=" + height + "\n";
    sidecar += "micron_per_pixel_x=" + calibratedMicronPerPixel(pixelWidth, unit) + "\n";
    sidecar += "micron_per_pixel_y=" + calibratedMicronPerPixel(pixelHeight, unit) + "\n";
    sidecar += "xml_path=" + xmlPath + "\n";
    File.saveString(sidecar, metaPath);

    showMessage(
        "Annotation Prepared",
        "Saved TIFF: " + tifPath +
        "\nSaved PNG: " + pngPath +
        "\nMetadata sidecar: " + metaPath +
        "\n\nNext step in Terminal:\npython3 src/annotation_workflow/annotation_tools.py finalize"
    );
}

function nextSampleNumber(dir) {
    list = getFileList(dir);
    maxNumber = 0;
    for (i = 0; i < list.length; i++) {
        name = list[i];
        if (!endsWith(name, ".tif")) continue;
        if (!startsWith(name, "sample_")) continue;
        dot = lastIndexOf(name, ".");
        stem = substring(name, 0, dot);
        parts = split(stem, "_");
        if (parts.length < 2) continue;
        value = parseInt(parts[1]);
        if (value > maxNumber) maxNumber = value;
    }
    return maxNumber + 1;
}

function isoDate() {
    getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, millisecond);
    month = month + 1;
    return "" + year + "-" + pad2(month) + "-" + pad2(dayOfMonth);
}

function pad2(value) {
    if (value < 10) return "0" + value;
    return "" + value;
}

function calibratedMicronPerPixel(value, unit) {
    lowerUnit = toLowerCase(unit);
    if (lowerUnit == "micron" || lowerUnit == "microns" || lowerUnit == "um" || unit == "µm")
        return "" + value;
    if (lowerUnit == "nm")
        return "" + (value / 1000.0);
    if (lowerUnit == "mm")
        return "" + (value * 1000.0);
    return "" + value;
}
