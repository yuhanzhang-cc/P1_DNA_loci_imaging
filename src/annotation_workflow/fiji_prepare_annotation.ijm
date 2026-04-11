// Fiji / ImageJ Macro
// 用途：
// 1. 用户先手动打开一个待标注图像或一个 .mvd2 的单个 series。
// 2. 用户手动切到合适的 frame。
// 3. 运行本宏。
// 4. 宏根据当前窗口生成单张 2D 的 sample_N 图像，保存 tif，打开 TrackMate。
// 5. 用户在 TrackMate 里手动调参数并手动保存 XML。
// 6. 关闭 TrackMate 后，宏继续保存 png 和 sidecar 元数据。

macro "Prepare Annotation [w]" {
    // 如果当前没有图像窗口，直接退出。
    if (nImages == 0) exit("No image is open. Please open one image or one .mvd2 series first.");

    // 仓库根目录。若仓库路径变化，需要修改这里。
    repoRoot = "/Users/mac/Developer/P1_DNA_loci_imaging";

    // 原始导出目录：单帧 tif。
    rawTifDir = repoRoot + "/data/raw/tif/";

    // 原始导出目录：带圈 png。
    rawPngDir = repoRoot + "/data/raw/png/";

    // TrackMate XML 目录。这里的 xml 仍需你在 TrackMate 里手动保存。
    trackmateXmlDir = repoRoot + "/data/raw/logs/trackmate_xml/";

    // sidecar 元数据目录。宏会自动写入。
    stateDir = repoRoot + "/data/raw/logs/annotation_state/";

    // 确保输出目录存在。
    File.makeDirectory(rawTifDir);
    File.makeDirectory(rawPngDir);
    File.makeDirectory(trackmateXmlDir);
    File.makeDirectory(stateDir);

    // 记录当前原始图像窗口的标题。
    originalTitle = getTitle();

    // 记录当前原始图像窗口的 ID，后续尽量用 ID 而不是标题操作窗口，减少改名带来的问题。
    originalId = getImageID();

    // 确保当前选中的仍然是原始图像窗口。
    selectImage(originalId);

    // 读取当前图像的维度信息。
    getDimensions(width, height, channels, slices, frames);

    // 读取当前所处位置：channel / slice / frame。
    Stack.getPosition(currentChannel, currentSlice, currentFrame);

    // 读取当前图像标定。
    getPixelSize(unit, pixelWidth, pixelHeight, voxelDepth);

    // 猜测 source_path。对于普通 tif/nd2 往往能猜到；猜不到就留空让你手动补。
    sourcePathGuess = guessSourcePath(originalTitle);

    // 猜测 source_file_name。默认用当前窗口标题。
    sourceFileNameGuess = guessSourceFileName(originalTitle);

    // 如果有 z stack，默认勾选 Max Intensity 投影；否则默认不勾选。
    defaultApplyProjection = 0;
    if (slices > 1) defaultApplyProjection = 1;

    // 弹出唯一的元数据对话框。
    // 这里不再要求你重新“选择图片文件”，只让你检查/修改元数据文本。
    Dialog.create("Annotation Metadata");
    Dialog.addMessage("请先确认：当前窗口已经是你要处理的图像/series，并且已经切到合适的 frame。");
    Dialog.addString("source_path", sourcePathGuess, 80);
    Dialog.addString("source_file_name", sourceFileNameGuess, 80);
    Dialog.addNumber("channel_index", currentChannel);
    Dialog.addCheckbox("apply_max_intensity_z_projection", defaultApplyProjection);
    Dialog.show();

    // 读取对话框结果。
    sourcePath = Dialog.getString();
    sourceFileName = Dialog.getString();
    selectedChannel = floor(Dialog.getNumber());
    doMaxProjection = Dialog.getCheckbox();

    // source_path 不能为空。
    if (trim(sourcePath) == "") {
        exit("source_path is empty. Please rerun the macro and fill the source_path field.");
    }

    // source_path 扩展名必须是支持的原始格式。
    if (!isSupportedSourcePath(sourcePath)) {
        exit("source_path must end with .tif, .tiff, .nd2, or .mvd2");
    }

    // source_file_name 为空时，回退到窗口标题。
    if (trim(sourceFileName) == "") sourceFileName = sourceFileNameGuess;

    // channel 序号必须合法。
    if (selectedChannel < 1 || selectedChannel > channels) {
        exit("Selected channel is out of range. Valid range: 1-" + channels);
    }

    // 没有 z stack 时，不允许投影。
    if (slices <= 1) doMaxProjection = 0;

    // 对 .mvd2 且做了 Max Intensity 的情况，在 source_file_name 前面自动加 MAX_。
    if (isMvd2Path(sourcePath) && doMaxProjection && !startsWith(sourceFileName, "MAX_")) {
        sourceFileName = "MAX_" + sourceFileName;
    }

    // 计算下一个 sample 编号。
    sampleNumber = nextSampleNumber(rawTifDir);

    // 统一的样本基础名。
    sampleBase = "sample_" + sampleNumber;

    // 目标 tif 路径。
    tifPath = rawTifDir + sampleBase + ".tif";

    // 目标 png 路径。
    pngPath = rawPngDir + sampleBase + ".png";

    // TrackMate XML 目标路径。你需要在 TrackMate 里手动保存到这里。
    xmlPath = trackmateXmlDir + sampleBase + ".xml";

    // sidecar 路径。
    metaPath = stateDir + sampleBase + ".meta.txt";

    // 根据当前窗口状态构造真正的 2D 标注图。
    preparedId = buildPreparedImage(
        originalId,
        sampleBase,
        selectedChannel,
        currentSlice,
        currentFrame,
        slices,
        doMaxProjection
    );

    // 切到导出的 2D 图。
    selectImage(preparedId);

    // 读取导出图的维度。
    getDimensions(outputWidth, outputHeight, outputChannels, outputSlices, outputFrames);

    // 如果不是单张 2D 图，直接报错退出。
    if (outputChannels != 1 || outputSlices != 1 || outputFrames != 1) {
        exit(
            "Prepared image is not a single 2D image.\n" +
            "channels=" + outputChannels +
            ", slices=" + outputSlices +
            ", frames=" + outputFrames
        );
    }

    // 保存 tif。
    saveAs("Tiff", tifPath);

    // 打开 TrackMate。
    run("TrackMate");

    // IJM 中 run("TrackMate") 不会稳定阻塞到窗口关闭。
    // 所以这里显式暂停，等你完成 TrackMate 操作、保存 XML、关闭 TrackMate 后再继续。
    waitForUser(
        "TrackMate 完成确认",
        "请在 TrackMate 中完成手动调参，并手动保存 XML 到：\n\n" + xmlPath +
        "\n\n请确认 TrackMate 已关闭，且图像上仍可见标点 overlay 后，再点击 OK。"
    );

    // TrackMate 关闭后，重新切回 prepared 图。
    selectImage(preparedId);

    // 保存带 TrackMate 标记的 png。
    exportAnnotatedPng(preparedId, pngPath);

    // z_axis 的记录逻辑：
    // 1. 做了投影 -> 记 0
    // 2. 原图本身没有 z -> 记 0
    // 3. 没投影但有 z -> 记当前 slice
    zAxisValue = 0;
    if (slices > 1 && !doMaxProjection) zAxisValue = currentSlice;

    // 写入当前日期。
    currentDate = isoDate();

    // 组装 sidecar 文本。
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
    sidecar += "x_pixel=" + outputWidth + "\n";
    sidecar += "y_pixel=" + outputHeight + "\n";
    sidecar += "micron_per_pixel_x=" + calibratedMicronPerPixel(pixelWidth, unit) + "\n";
    sidecar += "micron_per_pixel_y=" + calibratedMicronPerPixel(pixelHeight, unit) + "\n";
    sidecar += "xml_path=" + xmlPath + "\n";

    // 保存 sidecar。
    File.saveString(sidecar, metaPath);

}

// 根据原始图像构造真正的单张 2D 图像。
function buildPreparedImage(originalId, sampleBase, selectedChannel, currentSlice, currentFrame, slices, doMaxProjection) {
    // 回到原图。
    selectImage(originalId);

    // 切到用户指定的 C/Z/T 位置。
    Stack.setPosition(selectedChannel, currentSlice, currentFrame);

    // 如果需要做 z projection：
    if (doMaxProjection && slices > 1) {
        // 先提取“当前 frame + 选中 channel 的整个 z stack”。
        run(
            "Make Substack...",
            "channels=" + selectedChannel +
            " slices=1-" + slices +
            " frames=" + currentFrame
        );

        // Substack 产生的新窗口会成为当前窗口。
        zStackId = getImageID();
        rename(sampleBase + "_zstack");

        // 对这个 z stack 做 Max Intensity 投影。
        run("Z Project...", "projection=[Max Intensity]");

        // Z Project 产生的新窗口会成为当前窗口。
        projectionId = getImageID();

        // 把投影结果改名为 sample_N。
        rename(sampleBase);

        // 关闭中间 z stack。
        selectImage(zStackId);
        close();

        // 返回投影结果窗口 ID。
        selectImage(projectionId);
        return projectionId;
    }

    // 如果不做 z projection：
    // 只复制当前显示的单平面。
    // 注意：这里故意不传 "duplicate" 参数，因为该参数会勾选 Duplicate stack，
    // 从而把整段 time series 复制出来，导致 frame 又回到错误的 stack 行为。
    Stack.setPosition(selectedChannel, currentSlice, currentFrame);
    run("Duplicate...", "title=" + sampleBase);

    // Substack 产生的新窗口会成为当前窗口。
    preparedId = getImageID();
    rename(sampleBase);

    // 返回单平面图像窗口 ID。
    return preparedId;
}

// 估计 source_path。
function guessSourcePath(originalTitle) {
    // 先尝试读取当前图像所在目录。
    imageDir = getDirectory("image");

    // 如果拿不到目录，返回空字符串，让用户手动填写。
    if (imageDir == "") return "";

    // 对 .mvd2 series，窗口标题通常形如：
    // xxx.mvd2 - series_name
    // 这里优先从标题中截取真正的 .mvd2 文件名。
    mvd2Index = indexOf(originalTitle, ".mvd2");
    if (mvd2Index >= 0) {
        guessed = imageDir + substring(originalTitle, 0, mvd2Index + 5);
        if (isSupportedSourcePath(guessed)) return guessed;
    }

    // 对 .nd2，窗口标题有时也可能带后缀说明，优先截到 .nd2 为止。
    nd2Index = indexOf(originalTitle, ".nd2");
    if (nd2Index >= 0) {
        guessed = imageDir + substring(originalTitle, 0, nd2Index + 4);
        if (isSupportedSourcePath(guessed)) return guessed;
    }

    // 对 tiff，优先截到 .tif/.tiff 为止。
    tiffIndex = indexOf(originalTitle, ".tiff");
    if (tiffIndex >= 0) {
        guessed = imageDir + substring(originalTitle, 0, tiffIndex + 5);
        if (isSupportedSourcePath(guessed)) return guessed;
    }
    tifIndex = indexOf(originalTitle, ".tif");
    if (tifIndex >= 0) {
        guessed = imageDir + substring(originalTitle, 0, tifIndex + 4);
        if (isSupportedSourcePath(guessed)) return guessed;
    }

    // 尝试把完整标题直接拼成路径。
    guessed = imageDir + originalTitle;

    // 如果标题本身就是支持的文件名，就直接返回。
    if (isSupportedSourcePath(guessed)) return guessed;

    // 否则仅返回目录，方便用户手动补最后一段。
    return imageDir;
}

// 估计 source_file_name。默认就是当前窗口标题。
function guessSourceFileName(originalTitle) {
    return originalTitle;
}

// 根据已有 sample_N.tif 计算下一个编号。
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

// 生成 YYYY-MM-DD 日期。
function isoDate() {
    getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, millisecond);
    month = month + 1;
    return "" + year + "-" + pad2(month) + "-" + pad2(dayOfMonth);
}

// 两位数补零。
function pad2(value) {
    if (value < 10) return "0" + value;
    return "" + value;
}

// 将像素标定统一换算成 micron / pixel。
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

// 判断 source_path 是否是支持的原始文件扩展名。
function isSupportedSourcePath(path) {
    lowerPath = toLowerCase(path);
    if (endsWith(lowerPath, ".tif")) return true;
    if (endsWith(lowerPath, ".tiff")) return true;
    if (endsWith(lowerPath, ".nd2")) return true;
    if (endsWith(lowerPath, ".mvd2")) return true;
    return false;
}

// 判断是否是 .mvd2。
function isMvd2Path(path) {
    return endsWith(toLowerCase(path), ".mvd2");
}

// 导出带 overlay 的 PNG。
function exportAnnotatedPng(imageId, pngPath) {
    selectImage(imageId);

    // 如果 TrackMate 已经把 spots 作为 overlay 画到图上，
    // 先 flatten 再保存，这样 PNG 中会保留标点圈。
    if (Overlay.size > 0) {
        run("Flatten");
        flattenedId = getImageID();
        saveAs("PNG", pngPath);
        close();
        selectImage(imageId);
        return;
    }

    // 没有 overlay 时退回普通保存。
    saveAs("PNG", pngPath);
}
