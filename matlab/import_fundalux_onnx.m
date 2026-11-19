function import_fundalux_onnx(imageFile, modelDir, member)
% IMPORT_FUNDALUX_ONNX  Run one trained DR network in MATLAB and show Grad-CAM.
%
%   import_fundalux_onnx('out/enhanced.png')
%   import_fundalux_onnx('out/enhanced.png', '../models', 3)
%
% UNTESTED: written without access to MATLAB.
%
% The classifier is an ensemble; `member` picks which network to import
% (default 1). The app averages the members' probabilities, and the decision
% threshold in config.json applies to that average, not to a single member.
%
% The image must already be cropped and enhanced by the Python pipeline:
%   python -m fundalux.cli photo.jpg --save-dir out
% The network takes RGB values 0-255 (normalisation is inside the network) and
% returns 6 numbers: the referable-DR logit followed by 5 grade logits.

    if nargin < 2, modelDir = fullfile('..', 'models'); end
    if nargin < 3, member = 1; end
    cfg = jsondecode(fileread(fullfile(modelDir, 'config.json')));
    spec = cfg.dr.models(member);
    net = importNetworkFromONNX(fullfile(modelDir, strrep(spec.weights, '.pt', '.onnx')), ...
                                InputDataFormats = "BCSS");

    img = imread(imageFile);
    img = imresize(img, [spec.input_size spec.input_size]);
    X = dlarray(single(img), "SSCB");

    logits = extractdata(predict(net, X));
    pReferable = 1 / (1 + exp(-logits(1)));
    gradeProb = exp(logits(2:6) - max(logits(2:6)));
    gradeProb = gradeProb / sum(gradeProb);
    [~, grade] = max(gradeProb);

    fprintf('%s: P(referable DR) = %.3f (ensemble threshold %.3f)\n', spec.arch, ...
            pReferable, cfg.dr.threshold);
    fprintf('Estimated grade: %s\n', cfg.grade_names{grade});

    % Grad-CAM for the referable-DR output (the first of the six outputs).
    map = gradCAM(net, X, @(scores) scores(1));
    figure;
    imshow(img); hold on;
    imagesc(map, 'AlphaData', 0.5); colormap jet; hold off;
    title(sprintf('Grad-CAM, P(referable DR) = %.2f', pReferable));
end
