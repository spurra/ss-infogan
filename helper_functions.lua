-------------------------------------------------
--
--  HELPER FUNCTIONS
--
-------------------------------------------------

-- Constructs the target for the discriminator based on the generators input
-- contained in 'tensor'. The flag indicates if it should construct it for the
-- semi-supervised or unsupervised head.
function salient_input_to_target(tensor, semsup)
    local toReturn = {}

    if semsup then
        local categorical = tensor:narrow(2, 1, 10)
        local _, max_indices = categorical:max(2)
        toReturn = max_indices:typeAs(tensor):clone()
    else
        for i = 2, #n_classes_per_lc do
            local cat = tensor:narrow(2, c_offset[i], n_classes_per_lc[i])
            local _, max_idx = cat:max(2)
            max_idx = max_idx:typeAs(tensor):clone()

            toReturn[#toReturn+1] = max_idx
        end

        if n_cont_vars > 0 then
            toReturn[#toReturn+1] = tensor:narrow(2, c_offset[#c_offset], n_cont_vars):clone()
        end
    end

    return toReturn
end

-- Populates `res` such that each row contains a random one-hot vector.
-- That is, each row will be almost full of 0s, except for a 1 in a random
-- position.
function random_one_hot(res)
    local batch_size = res:size(1)
    local n_categories = res:size(2)

    local probabilities = res.new(n_categories):fill(1 / n_categories)
    local indices = torch.multinomial(probabilities, batch_size, true):view(-1, 1)

    res:zero():scatter(2, indices, 1)
end


-- Test input for G to test quality of its samples with respect to C
local G_test_input = {}
local n_classes = n_classes_per_lc[1]
-- Construct the input so that batch_size samples of each class is created
for i = 1, n_classes do
    G_test_input[i] = torch.CudaTensor(batch_size, n_gen_inputs):zero()
    G_test_input[i]:narrow(2, n_salient_vars + 1, n_noise_vars):normal(0, 1)
    G_test_input[i][{{}, {i}}] = torch.ones(batch_size)

    for j=2, #n_classes_per_lc do
        random_one_hot(G_test_input[i]:narrow(2, c_offset[j], n_classes_per_lc[j]))
    end
    if n_cont_vars > 0 then
        G_test_input[i]:narrow(2, n_cat_vars + 1, n_cont_vars):uniform(-1, 1)
    end
end

function test_signal(G, C)
    -- Create the synthetic samples
    G:evaluate()
    C1:evaluate()
    local synthSamples
    local predictions = torch.CudaTensor(n_classes * batch_size, n_classes_per_lc[1])
    for i = 1, n_classes do
        local range1 = (i - 1) * batch_size + 1
        local range2 = i * batch_size
        synthSamples = G:forward(G_test_input[i])
        predictions[{{range1, range2}, {}}] = C1:forward(synthSamples)
    end

    local meanPred = torch.Tensor(n_classes, n_classes):zero()

    for i = 1, n_classes do
        range1 = (i-1)*batch_size + 1
        range2 = i*batch_size
        -- Get the probabilities in [0,1] range
        meanPred[{{i}, {}}] = predictions[{{range1, range2}, {}}]:mean(1):squeeze():float()
    end

    -- Get a greedy bijective mapping from salient class to proper class.
    -- mapping(salient_class) = proper_class
    local mapping = torch.Tensor(n_classes):zero()
    for i = 1, n_classes do
        -- Find maximum probability value
        local r_max_val, r_max_idx = meanPred:max(2)
        local c_max_val, c_max_idx = r_max_val:max(1)
        local row_idx = c_max_idx[1][1]
        local col_idx = r_max_idx[row_idx][1]
        local max_val = meanPred[row_idx][col_idx]
        mapping[row_idx] = col_idx
        -- Assign the salient class mapped to -1 so it does not get mapped again.
        meanPred[{{row_idx}, {}}] = -1
        -- Assign the proper class mapped to -1 so it does not get mapped to again.
        meanPred[{{}, {col_idx}}] = -1
    end

    local loss = torch.Tensor(n_classes):zero()
    for i = 1, n_classes do
        range1 = (i-1)*batch_size + 1
        range2 = i*batch_size

        -- Compute the mean 0-1 loss for each salient class i.
        local _, predClass = predictions[{{range1, range2}, {}}]:max(2)
        loss[i] = predClass:ne(mapping[i]):float():mean()
    end

    return loss:mean()
end

function evaluate_infoHead_real(D)
    local predictions = D:forward(dataset.test_inputs)[2]
    local meanPred = torch.Tensor(n_classes, n_classes):zero()

    for i = 1, n_classes do
        range1 = (i-1)*batch_size + 1
        range2 = i*batch_size
        -- Get the probabilities in [0,1] range
        meanPred[{{i}, {}}] = predictions[{{range1, range2}, {}}]:mean(1):squeeze():float()
    end

    -- Get a greedy bijective mapping from salient class to proper class.
    -- mapping(proper_class) = salient_class
    local mapping = torch.CudaTensor(n_classes):zero()
    for i = 1, n_classes do
        -- Find maximum probability value
        local r_max_val, r_max_idx = meanPred:max(2)
        local c_max_val, c_max_idx = r_max_val:max(1)
        local row_idx = c_max_idx[1][1]
        local col_idx = r_max_idx[row_idx][1]
        local max_val = meanPred[row_idx][col_idx]
        mapping[row_idx] = col_idx
        -- Assign the salient class mapped to -1 so it does not get mapped again.
        meanPred[{{row_idx}, {}}] = -1
        -- Assign the proper class mapped to -1 so it does not get mapped to again.
        meanPred[{{}, {col_idx}}] = -1
    end

    local loss = torch.Tensor(n_classes):zero()
    for i = 1, n_classes do
        range1 = (i-1)*batch_size + 1
        range2 = i*batch_size

        local _, predClass = predictions[{{range1, range2}, {}}]:max(2)
        loss[i] = predClass:ne(mapping[i]):float():mean()
    end

    return loss:mean()
end

function evaluate_infoHead_fake(G, D)
    G:evaluate()

    -- Create the synthetic samples
    local synthSamples = torch.CudaTensor(n_classes * batch_size, samples_dim[1], samples_dim[2], samples_dim[3])
    for i = 1, n_classes do
      local range1 = (i - 1) * batch_size + 1
      local range2 = i * batch_size
      synthSamples[{{range1, range2}, {}}] = G:forward(G_test_input[i])
    end

    -- Evaluate the synthetic samples on the classifier
    local predictions = D:forward(synthSamples)
    local pred_semsup = predictions[2]
    local pred_unsup = predictions[3]
    local loss_semsup = torch.Tensor(n_classes):zero()
    local loss_unsup = torch.Tensor(n_classes):zero()

    for i = 1, n_classes do
        range1 = (i-1)*batch_size + 1
        range2 = i*batch_size

        -- -- Compute the mean 0-1 loss for each salient class i.
        local true_c = salient_input_to_target(G_test_input[i], true)

        -- Compute the mean 0-1 loss for each salient class i.
        local _, predClass = pred_semsup[{{range1, range2}, {}}]:max(2)
        loss_semsup[i] = predClass:long():ne(true_c:long()):float():mean()

        true_c = salient_input_to_target(G_test_input[i], false)

        -- Construct the output
        if n_unsup_vars > 0 then
            local tmp_output = {}
            for j = 1, #pred_unsup do
                tmp_output[j] = pred_unsup[j][{{range1, range2}, {}}]
            end

            loss_unsup[i] = info_head_unsup_crit:forward(tmp_output, true_c)
        end
    end

    return loss_semsup:mean(), loss_unsup:mean()
end
