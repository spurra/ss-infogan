--[[
    This code is a heavily modified version of and based on the code found in

    https://github.com/anibali/infogan
]]--

require('torch')    -- Essential Torch utilities
require('image')    -- Torch image handling
require('nn')       -- Neural network building blocks
require('optim')    -- Optimisation algorithms
require('cutorch')  -- 'torch' on the GPU
require('cunn')     -- 'nn' on the GPU
require('cudnn')    -- Torch bindings to CuDNN
require 'gnuplot'

local pl = require('pl.import_into')()
local tnt = require('torchnet')
local nninit = require('nninit')
-- TODO Change once code is tested
local dataset = require('reproducable_research/dataset')
dbg = require 'reproducable_research/debugger'


--- OPTIONS ---

local params = {}
-- 500 is 50'000 iterations
local n_epochs = 100
local n_updates_per_epoch = 100
local batch_size = 128
local disc_learning_rate = 1e-4
local gen_learning_rate = 1e-3
local rng_seed = 1

-- Percentage of data to unlabel
local unlabelled_percentage = 0.0
local info_reg_sup_coef_learn = 1
local info_reg_sup_coef_train = 1
local info_reg_unsup_coef_train = 1
local info_reg_unsup_coef_learn = 1

n_classes_per_lc = {1, 10}
c_offset = torch.totable(torch.cumsum(torch.Tensor(n_classes_per_lc)))
table.remove(n_classes_per_lc, 1)
-- At this point, n_classes_per_lc[i] is the number of classes the i-th
-- categorical latent code contains.
-- c_offset[i] corresponds to the starting index of categorical code i.

-- Helper variables
local n_cat_vars = c_offset[#c_offset] - 1
local n_cont_vars = 2
local n_noise_vars = 62
local n_semsup_vars = n_classes_per_lc[1]
local n_unsup_vars = n_cat_vars - n_semsup_vars + n_cont_vars
local n_gen_inputs = n_cont_vars + n_cat_vars + n_noise_vars
local n_salient_vars = n_cont_vars + n_cat_vars

-- f true, sets the unlabeled sampling probability to 0 and grows it linearly
-- to the true value. Useful if the number of labeled data is small.
local sampling_prob_growth = true
local additive_prob_growth = 0.1
local unlabelled_sampling_prob = unlabelled_percentage
-- If enabled, it will grow the unlabelled sampling probability in each epoch until the value of
-- unlabelled_percentage
if sampling_prob_growth and unlabelled_percentage < 1 then
  unlabelled_sampling_prob = 0.0
end

-- Instance noise parameters
local instance_noise_std = 0
local instance_noise_annealing = 0

local samples_dim = {1, 28, 28}

-- TODO Remove
local classifier_path = 'classifier_models/mnist_classifier_convnet.t7'
local dataset_train_path = 'data/mnist/mnist_train.t7'
local dataset_test_path = 'data/mnist/mnist_test.t7'

-- Save the current parameters
params.n_epochs = n_epochs
params.n_updates_per_epoch = n_updates_per_epoch
params.batch_size = batch_size
params.info_reg_sup_coef_learn = info_reg_sup_coef_learn
params.info_reg_sup_coef_train = info_reg_sup_coef_train
params.info_reg_unsup_coef_learn = info_reg_unsup_coef_learn
params.info_reg_unsup_coef_train = info_reg_unsup_coef_train
params.disc_learning_rate = disc_learning_rate
params.gen_learning_rate = gen_learning_rate
params.rng_seed = rng_seed
params.n_cont_vars = n_cont_vars
params.n_noise_vars = n_noise_vars
params.n_classes_per_lc = n_classes_per_lc
params.unlabelled_percentage = unlabelled_percentage
params.unlabelled_sampling_prob = unlabelled_sampling_prob
params.additive_prob_growth = additive_prob_growth
params.sampling_prob_growth = tostring(sampling_prob_growth)
params.instance_noise_std = instance_noise_std
params.instance_noise_annealing = instance_noise_annealing


--- INIT ---

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- Set manual seeds for reproducible RNG
torch.manualSeed(rng_seed)
cutorch.manualSeedAll(rng_seed)
math.randomseed(rng_seed)

torch.setdefaulttensortype('torch.FloatTensor')

local outFolder = 'exp1/'

i = 1
while file_exists(outFolder) do
  i = i + 1
  outFolder = 'exp' .. i .. '/'
end
print('Output directory: ' .. outFolder)
lfs.mkdir(outFolder)


local C1 = torch.load(classifier_path):cuda()
local train_data = dataset.new(dataset_train_path, unlabelled_percentage, dataset_test_path)
local train_iter_labelled = train_data:make_iterator(batch_size)
local train_iter_unlabelled = train_data:make_iterator_unlabelled(batch_size)

--- MODEL ---

Seq = nn.Sequential
ReLU = cudnn.ReLU

function SpatBatchNorm(n_outputs)
    return nn.SpatialBatchNormalization(n_outputs, 1e-5, 0.1)
        :init('weight', nninit.normal, 1.0, 0.02)
        :init('bias', nninit.constant, 0)
end

function BatchNorm(n_outputs)
    return nn.BatchNormalization(n_outputs, 1e-5, 0.1)
        :init('weight', nninit.normal, 1.0, 0.02)
        :init('bias', nninit.constant, 0)
end

function Conv(...)
    local conv = cudnn.SpatialConvolution(...)
        :init('weight', nninit.normal, 0.0, 0.02)
        :init('bias', nninit.constant, 0)
    conv:setMode(
        'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
        'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
        'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')
    return conv
end

function FullConv(...)
    local conv = cudnn.SpatialFullConvolution(...)
        :init('weight', nninit.normal, 0.0, 0.02)
        :init('bias', nninit.constant, 0)
    conv:setMode(
        'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM',
        'CUDNN_CONVOLUTION_BWD_DATA_ALGO_1',
        'CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1')
    return conv
end

function LeakyReLU(leakiness, in_place)
    leakiness = leakiness or 0.01
    in_place = in_place == nil and true or in_place
    return nn.LeakyReLU(leakiness, in_place)
end

function Linear(...)
    return nn.Linear(...)
        :init('weight', nninit.normal, 0.0, 0.02)
        :init('bias', nninit.constant, 0)
end

--- CRITERIA ---

local disc_head_criterion = nn.BCECriterion()
local info_head_semsup_crit = nn.CrossEntropyCriterion()
local info_head_unsup_crit = nn.ParallelCriterion()

--- MODEL ---

local generator = Seq()
    :add(Linear(n_gen_inputs, 1024))
    :add(BatchNorm(1024))
    :add(ReLU(true))
    :add(Linear(1024, 128 * 7 * 7))
    :add(BatchNorm(128 * 7 * 7))
    :add(ReLU(true))
    :add(nn.Reshape(128, 7, 7))
    :add(FullConv(128, 64, 4,4, 2,2, 1,1))
    :add(SpatBatchNorm(64))
    :add(ReLU(true))
    :add(FullConv(64, 1, 4,4, 2,2, 1,1))
    :add(nn.Sigmoid())

local discriminator_body = Seq()
    :add(Conv(1, 64, 4,4, 2,2, 1,1))
    :add(LeakyReLU())
    :add(Conv(64, 128, 4,4, 2,2, 1,1))
    :add(SpatBatchNorm(128))
    :add(LeakyReLU())
    :add(nn.Reshape(128 * 7 * 7))
    :add(Linear(128 * 7 * 7, 1024))
    :add(BatchNorm(1024))
    :add(LeakyReLU())

local discriminator_head = Seq()
    :add(Linear(1024, 1))
    :add(nn.Sigmoid())

local info_head_semsup = Seq()
    :add(Linear(1024, n_semsup_vars))

local discriminator = Seq()
    :add(discriminator_body)

local heads = nn.ConcatTable()
    :add(discriminator_head)
    :add(info_head_semsup)


local info_head_unsup = nil
if n_unsup_vars > 0 then
    info_head_unsup = Seq()
        :add(Linear(1024, 128))
        :add(BatchNorm(128))
        :add(LeakyReLU())
        :add(Linear(128, n_unsup_vars))

    local concat = nn.ConcatTable()
    for i=2,#n_classes_per_lc do
        concat:add(nn.Narrow(2, c_offset[i]-n_classes_per_lc[1], n_classes_per_lc[i]))
    end
    if n_cont_vars > 0 then
        concat:add(nn.Narrow(2, c_offset[#c_offset]-n_classes_per_lc[1], n_cont_vars))
    end

    info_head_unsup:add(concat)
    heads:add(info_head_unsup)

    for i = 2, #n_classes_per_lc do
        info_head_unsup_crit:add(nn.CrossEntropyCriterion())
    end
    if n_cont_vars > 0 then
        info_head_unsup_crit:add(nn.MSECriterion())
    end
end

discriminator:add(heads)

-- Run on the GPU
generator:cuda()
discriminator:cuda()
disc_head_criterion:cuda()
info_head_unsup_crit:cuda()
info_head_semsup_crit:cuda()

print(discriminator)
print(generator)

params.G = tostring(generator)
params.D = tostring(discriminator)

torch.save(outFolder .. 'params.t7', params)


-- LOGGING --
local log_text = require('torchnet.log.view.text')

local log_keys = {'epoch', 'fake_loss', 'info_loss_semsup','info_loss_unsup',
  'real_loss', 'gen_loss', 'time'}

local log = tnt.Log{
  keys = log_keys,
  onFlush = {
    log_text{
      keys = log_keys,
      format = {'epoch=%3d', 'fake_loss=%8.5f', 'info_loss_semsup=%8.6f', 'info_loss_unsup=%8.6f',
        'real_loss=%8.6f', 'gen_loss=%8.6f', 'time=%5.2fs'}
    }
  }
}

-- HELPER FUNCTIONS

-- Creates targets for the salient part of the generator input
local function salient_input_to_target(tensor, semsup)
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
local function random_one_hot(res)
  local batch_size = res:size(1)
  local n_categories = res:size(2)

  local probabilities = res.new(n_categories):fill(1 / n_categories)
  local indices = torch.multinomial(probabilities, batch_size, true):view(-1, 1)

  res:zero():scatter(2, indices, 1)
end


--- TRAIN ---



-- Test input for G to print
n_classes = n_classes_per_lc[1]
G_print_input = {}
nr_inputs_G_print_per_class = 12
for l = 1,15 do
    z = torch.Tensor(1, n_noise_vars):normal(0,1)
    G_print_input[l] = {}
    local rand_offset = {}
    local rand_cont = {}
    for i = 2, #n_classes_per_lc do
        rand_offset[i] = torch.random(0, n_classes_per_lc[i] - 1)
    end
    for i = 1, n_cont_vars do
        rand_cont[i] = torch.uniform(-1,1)
    end


    for k = 1, n_cont_vars do
        n_elem = n_classes * nr_inputs_G_print_per_class
        G_print_input[l][k] = torch.CudaTensor(n_elem, n_gen_inputs):zero()
        G_print_input[l][k][{{1, n_elem}, {n_salient_vars + 1, n_salient_vars + n_noise_vars}}] = z:repeatTensor(n_elem, 1)
        for i=1, n_classes do
            range1 = (i-1)*nr_inputs_G_print_per_class + 1
            range2 = i*nr_inputs_G_print_per_class
            G_print_input[l][k][{{range1, range2}, {i}}] = torch.ones(nr_inputs_G_print_per_class)
            G_print_input[l][k][{{range1, range2}, {n_cat_vars+k}}] = torch.linspace(-2,2, nr_inputs_G_print_per_class)
        end
        for i = 2, #n_classes_per_lc do
            G_print_input[l][k][{{}, {c_offset[i] + rand_offset[i]}}] = 1
        end
    end

    for k = n_cont_vars + 1, n_cont_vars + #n_classes_per_lc - 1 do
        idx = k - n_cont_vars + 1
        curr_n_class = n_classes_per_lc[idx]
        n_elem = n_classes * curr_n_class
        G_print_input[l][k] = torch.CudaTensor(n_elem, n_gen_inputs):zero()
        G_print_input[l][k][{{1, n_elem}, {n_salient_vars + 1, n_salient_vars + n_noise_vars}}] = z:repeatTensor(n_elem, 1)
        for i=1, n_classes do
            range1 = (i-1)*curr_n_class + 1
            range2 = i*curr_n_class
            G_print_input[l][k][{{range1, range2}, {i}}] = torch.ones(curr_n_class)
            for p = 0, curr_n_class - 1 do
                r1 = range1 + p
                r2 = c_offset[idx] + p
                G_print_input[l][k][{{r1}, {r2}}] = 1
            end
        end

        -- Fill the first class of the other categorical variables with 1
        for p = 2, idx - 1 do
            G_print_input[l][k][{{}, {c_offset[p]}}] = 1
        end
        for p = idx + 1, #n_classes_per_lc do
            G_print_input[l][k][{{}, {c_offset[p] + rand_offset[p]}}] = 1
        end

        for p = 1, n_cont_vars do
            G_print_input[l][k][{{}, {n_cat_vars + p}}] = rand_cont[p]
        end
    end

    lk = #G_print_input[l] + 1
    n_elem = n_classes * 10
    G_print_input[l][lk] = torch.CudaTensor(n_elem, n_gen_inputs):zero()
    G_print_input[l][lk][{{}, {n_salient_vars + 1, n_salient_vars + n_noise_vars}}] = z:repeatTensor(n_elem, 1)
    for i=1, n_classes do
        range1 = (i-1)*10 + 1
        range2 = i*10
        G_print_input[l][lk][{{range1, range2}, {i}}] = torch.ones(10)
    end
    for i = 2, #n_classes_per_lc do
        random_one_hot(G_print_input[l][lk]:narrow(2,c_offset[i], n_classes_per_lc[i]))
    end
    if n_cont_vars > 0 then
        G_print_input[l][lk]:narrow(2, n_cat_vars + 1, n_cont_vars):uniform(-1,1)
    end
end


-- Test input for G to test quality of its samples with respect to C
local G_test_input = {}
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


local real_input = torch.CudaTensor()
local gen_input = torch.CudaTensor()
local fake_input = torch.CudaTensor()
local disc_target = torch.CudaTensor()
local target = torch.CudaTensor()
local labelled_batch

local disc_params, disc_grad_params = discriminator:getParameters()
local gen_params, gen_grad_params = generator:getParameters()

local fake_loss_meter = tnt.AverageValueMeter()
local info_loss_semsup_meter = tnt.AverageValueMeter()
local info_loss_unsup_meter = tnt.AverageValueMeter()
local real_loss_meter = tnt.AverageValueMeter()
local gen_loss_meter = tnt.AverageValueMeter()
local time_meter = tnt.TimeMeter()

local mean_class_losses_1 = torch.Tensor(n_epochs)
local infoHead_performance = torch.Tensor(n_epochs, 3)
local gradient_norms = torch.Tensor(n_epochs, 3)

local function test_signal(G, C)
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

local function evaluate_infoHead_real(D)
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

local function evaluate_infoHead_fake(G, D)
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


-- Calculate outputs and gradients for the discriminator
local do_discriminator_step = function(new_params)
    if new_params ~= disc_params then
    disc_params:copy(new_params)
    end

    disc_grad_params:zero()

    local loss_real = 0
    local loss_fake = 0
    local loss_info = 0
    local loss_info_unsup = 0
    local loss_info_semsup_real = 0
    local loss_info_semsup_fake = 0

    local ri_size = real_input:size()

    -- Add instance noise to real input
    if (instance_noise_std > 0) then
        real_input:add(torch.Tensor(real_input:size()):typeAs(real_input):normal(0, instance_noise_std))
    end

    -- Train with real images (from dataset)
    local batch_size = real_input:size(1)
    disc_target:resize(batch_size, 1)
    local dbodyout = discriminator_body:forward(real_input)
    local dheadout = discriminator_head:forward(dbodyout)
    disc_target:fill(1)
    loss_real = disc_head_criterion:forward(dheadout, disc_target)
    local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
    local dloss_ddheadin = discriminator_head:backward(dbodyout, dloss_ddheadout)
    discriminator_body:backward(real_input, dloss_ddheadin)

    if labelled_batch then
        -- Train info_head_semsup with real labeled images
        local iheadout_semsup_real = info_head_semsup:forward(dbodyout)
        local info_target_semsup_real = target
        loss_info_semsup = info_head_semsup_crit:forward(iheadout_semsup_real, info_target_semsup_real) * info_reg_sup_coef_train
        local dloss_diheadout_semsup_real = info_head_semsup_crit:backward(iheadout_semsup_real, info_target_semsup_real)
        dloss_diheadout_semsup_real:mul(info_reg_sup_coef_train)
        local dloss_diheadin_semsup_real = info_head_semsup:backward(dbodyout, dloss_diheadout_semsup_real)
        discriminator_body:backward(real_input, dloss_diheadin_semsup_real)
    end

    gen_input:resize(ri_size[1], n_gen_inputs)
    -- Train with fake images (from generator)
    for i = 1, #n_classes_per_lc do
        random_one_hot(gen_input:narrow(2, c_offset[i], n_classes_per_lc[i]))
    end

    if n_cont_vars > 0 then
        gen_input:narrow(2, c_offset[#c_offset], n_cont_vars):uniform(-1, 1)
    end
    gen_input:narrow(2, n_salient_vars + 1, n_noise_vars):normal(0, 1)

    generator:forward(gen_input)
    fake_input:resizeAs(generator.output):copy(generator.output)

    -- Add instance noise to fake_input
    if (instance_noise_std > 0) then
        fake_input:add(torch.Tensor(fake_input:size()):typeAs(fake_input):normal(0, instance_noise_std))
    end

    local dbodyout = discriminator_body:forward(fake_input)
    -- Train discriminator head with fake images
    local dheadout = discriminator_head:forward(dbodyout)
    disc_target:fill(0)
    loss_fake = disc_head_criterion:forward(dheadout, disc_target)
    local dloss_ddheadout = disc_head_criterion:backward(dheadout, disc_target)
    local dloss_ddheadin = discriminator_head:backward(dbodyout, dloss_ddheadout)
    discriminator_body:backward(fake_input, dloss_ddheadin)


    if unlabelled_percentage == 1 then
        -- Train info_head_semsup with fake images
        local iheadout_semsup_fake = info_head_semsup:forward(dbodyout)
        local info_target_semsup_fake = salient_input_to_target(gen_input:narrow(2, 1, n_salient_vars), true)
        loss_info_semsup = info_head_semsup_crit:forward(iheadout_semsup_fake, info_target_semsup_fake) * info_reg_sup_coef_train
        local dloss_diheadout_semsup_real = info_head_semsup_crit:backward(iheadout_semsup_fake, info_target_semsup_fake)
        dloss_diheadout_semsup_real:mul(info_reg_sup_coef_train)
        local dloss_diheadin_semsup_fake = info_head_semsup:backward(dbodyout, dloss_diheadout_semsup_real)
        discriminator_body:backward(fake_input, dloss_diheadin_semsup_fake)
    end



    -- Train info_head_unsup with fake images
    loss_info_unsup = 0
    if info_head_unsup ~= nil then
        local iheadout_unsup = info_head_unsup:forward(dbodyout)
        local info_target_unsup = salient_input_to_target(gen_input:narrow(2, 1, n_salient_vars), false)
        loss_info_unsup = info_head_unsup_crit:forward(iheadout_unsup, info_target_unsup) * info_reg_unsup_coef_train
        local dloss_diheadout_unsup = info_head_unsup_crit:backward(iheadout_unsup, info_target_unsup)
        for i, t in ipairs(dloss_diheadout_unsup) do
            t:mul(info_reg_unsup_coef_train)
        end
        local dloss_diheadin_unsup = info_head_unsup:backward(dbodyout, dloss_diheadout_unsup)
        discriminator_body:backward(fake_input, dloss_diheadin_unsup)
    end

    loss_info = loss_info_unsup + loss_info_semsup

    -- Update average value meters
    real_loss_meter:add(loss_real)
    fake_loss_meter:add(loss_fake)
    info_loss_semsup_meter:add(loss_info_semsup_real)
    info_loss_unsup_meter:add(loss_info_unsup)

    -- Calculate combined loss
    local loss = loss_real + loss_fake + loss_info

    -- Anneal instance noise
    instance_noise_std = instance_noise_std * instance_noise_annealing

    return loss, disc_grad_params
end

-- Calculate outputs and gradients for the generator
local do_generator_step = function(new_params)
    if new_params ~= gen_params then
        gen_params:copy(new_params)
    end

    gen_grad_params:zero()

    disc_target:fill(1)
    fake_input:resizeAs(generator.output):copy(generator.output)

    local dbodyout = discriminator_body:forward(fake_input)
    -- Train discriminator head with fake images
    local dheadout = discriminator_head:forward(dbodyout)

    -- Disc_head
    gen_loss = disc_head_criterion:forward(dheadout, disc_target)
    local dloss_ddheadout = disc_head_criterion:updateGradInput(dheadout, disc_target)
    local dloss_ddheadin = discriminator_head:updateGradInput(dbodyout, dloss_ddheadout)

    dloss_dgout = discriminator_body:updateGradInput(fake_input, dloss_ddheadin)
    gen_loss_meter:add(gen_loss)
    generator:backward(gen_input, dloss_dgout)

    -- Semsup_head
    local iheadout_semsup_fake = info_head_semsup:forward(dbodyout)
    local info_target_semsup_fake = salient_input_to_target(gen_input:narrow(2, 1, n_salient_vars), true)
    loss_info_semsup_fake = info_head_semsup_crit:forward(iheadout_semsup_fake, info_target_semsup_fake) * info_reg_sup_coef_learn
    local dloss_diheadout_semsup_fake = info_head_semsup_crit:updateGradInput(iheadout_semsup_fake, info_target_semsup_fake)
    dloss_diheadout_semsup_fake:mul(info_reg_sup_coef_learn)
    local dloss_diheadin_semsup_fake = info_head_semsup:updateGradInput(dbodyout, dloss_diheadout_semsup_fake)

    -- Unsup_head
    local dloss_diheadin_unsup = torch.zeros(dloss_ddheadin:size()):typeAs(dloss_ddheadin)
    if info_head_unsup ~= nil then
        local iheadout_unsup = info_head_unsup:forward(dbodyout)
        local info_target_unsup = salient_input_to_target(gen_input:narrow(2, 1, n_salient_vars), false)
        loss_info_unsup = info_head_unsup_crit:forward(iheadout_unsup, info_target_unsup) * info_reg_unsup_coef_learn
        local dloss_diheadout_unsup = info_head_unsup_crit:updateGradInput(iheadout_unsup, info_target_unsup)
        for i, t in ipairs(dloss_diheadout_unsup) do
            t:mul(info_reg_unsup_coef_learn)
        end
        dloss_diheadin_unsup = info_head_unsup:updateGradInput(dbodyout, dloss_diheadout_unsup)
    end

    dloss_dgout = discriminator_body:updateGradInput(fake_input, dloss_diheadin_semsup_fake + dloss_diheadin_unsup)
    gen_loss_meter:add(gen_loss)
    generator:backward(gen_input, dloss_dgout)

    return gen_loss, gen_grad_params
end


-- Discriminator optimiser
local disc_optimiser = {
        method = optim.adam,
        config = {
        epsilon = 1e-8,
        learningRate = disc_learning_rate,
        beta1 = 0.5
    },
    state = {}
}

-- Generator optimiser
local gen_optimiser = {
        method = optim.adam,
        config = {
        epsilon = 1e-8,
        learningRate = gen_learning_rate,
        beta1 = 0.5
    },
    state = {}
}

-- Takes a tensor containing many images and uses them as tiles to create one
-- big image. Assumes row-major order.
local function tile_images(images, rows, cols)
    local tiled = torch.Tensor(images:size(2), images:size(3) * rows, images:size(4) * cols)
    tiled:zero()
    for i = 1, math.min(images:size(1), rows * cols) do
    local col = (i - 1) % cols
        local row = math.floor((i - 1) / cols)
        tiled
            :narrow(2, row * images:size(3) + 1, images:size(3))
            :narrow(3, col * images:size(4) + 1, images:size(4))
            :copy(images[i])
    end
    return tiled
end

local iter_inst_labelled = train_iter_labelled()
local iter_inst_unlabelled = train_iter_unlabelled()

fake_losses = {}
info_losses_semsup = {}
info_losses_unsup = {}
real_losses = {}
gen_losses = {}

curr_epoch = 0
-- Training loop
for epoch = 1, n_epochs do
    fake_loss_meter:reset()
    info_loss_semsup_meter:reset()
    info_loss_unsup_meter:reset()
    real_loss_meter:reset()
    gen_loss_meter:reset()
    time_meter:reset()

    curr_epoch = epoch

    local n_labelled_sampled = 0
    local n_unlabelled_sampled = 0

    -- Do training iterations for the epoch
    for iteration = 1, n_updates_per_epoch do

        local sample
        -- Biased coin toss to decide if we sampling from labelled or unlabelled data
        if torch.bernoulli(unlabelled_sampling_prob) == 0 then
            sample = iter_inst_labelled()
            n_labelled_sampled = n_labelled_sampled + 1
            labelled_batch = true
        else
            sample = iter_inst_unlabelled()
            n_unlabelled_sampled = n_unlabelled_sampled + 1
            labelled_batch = false
        end

        if not sample or sample.input:size(1) < batch_size then
            -- Restart iterator
            if labelled_batch then
                iter_inst_labelled = train_iter_labelled()
                sample = iter_inst_labelled()
            else
                iter_inst_unlabelled = train_iter_unlabelled()
                sample = iter_inst_unlabelled()
            end
        end

        -- Copy real inputs from the dataset onto the GPU
        input = sample.input
        real_input:resize(input:size()):copy(input)
        target:resize(sample.target:size()):copy(sample.target)

        -- Update the discriminator network
        disc_optimiser.method(
            do_discriminator_step,
            disc_params,
            disc_optimiser.config,
            disc_optimiser.state
        )
        -- Update the generator network
        gen_optimiser.method(
            do_generator_step,
            gen_params,
            gen_optimiser.config,
            gen_optimiser.state
        )
    end

    print('unlabelled_sampling_prob: ' .. unlabelled_sampling_prob .. ' n_labelled_sampled: ' .. n_labelled_sampled .. ' n_unlabelled_sampled: ' .. n_unlabelled_sampled)

    if sampling_prob_growth then
        -- Grow the unlabelled sampling probability
        if unlabelled_sampling_prob < unlabelled_percentage then
            unlabelled_sampling_prob = unlabelled_sampling_prob + additive_prob_growth
        end
        if unlabelled_sampling_prob > unlabelled_percentage then
            unlabelled_sampling_prob = unlabelled_percentage
        end
    end

    generator:evaluate()
    fake_images = {}

    for i = 1, #G_print_input[1] do
        local n_elem = G_print_input[1][i]:size(1) / n_classes_per_lc[1]
        local curr_fake_image = torch.Tensor(15, samples_dim[1], samples_dim[2] * n_classes_per_lc[1], samples_dim[3] * n_elem)
        for l=1,15 do
            curr_fake_image[l] = tile_images(generator:forward(G_print_input[l][i]):float(), n_classes_per_lc[1], n_elem)
        end
        fake_images[i] = tile_images(curr_fake_image, 3, 5)
    end

    infoHead_performance[{{epoch}, {1}}] = evaluate_infoHead_real(discriminator)
    infoHead_performance[{{epoch}, {2}}], infoHead_performance[{{epoch}, {3}}] = evaluate_infoHead_fake(generator, discriminator)
    mean_class_losses_1[epoch] = test_signal(generator, C1)

    discriminator:training()
    generator:training()

  -- Update log
    log:set{
        epoch = epoch,
        fake_loss = fake_loss_meter:value(),
        info_loss_semsup = mean_class_losses_1[epoch],
        info_loss_unsup = info_loss_unsup_meter:value(),
        real_loss = real_loss_meter:value(),
        gen_loss = gen_loss_meter:value(),
        time = time_meter:value()
    }
    log:flush()

    -- Save outputs
    info_losses_semsup[#info_losses_semsup+1] = info_loss_semsup_meter:value()
    info_losses_unsup[#info_losses_unsup+1] = info_loss_unsup_meter:value()
    real_losses[#real_losses+1] = real_loss_meter:value()
    gen_losses[#gen_losses+1] = gen_loss_meter:value()
    fake_losses[#fake_losses+1] = fake_loss_meter:value()

    local model_dir = pl.path.join(outFolder, 'models')
    pl.dir.makepath(model_dir)

    discriminator:clearState()
    local model_disc_file = pl.path.join(model_dir, 'infogan_svhn_disc.t7')
    torch.save(model_disc_file, discriminator)

    generator:clearState()
    local model_gen_file = pl.path.join(model_dir, 'infogan_svhn_gen.t7')
    torch.save(model_gen_file, generator)

    local image_dir = pl.path.join(outFolder, 'images')
    pl.dir.makepath(image_dir)

    if epoch % 10 == 0 then
        for i = 1, #fake_images do
            local image_basename = string.format('fake_images_%d_%04d.png', i, epoch)
            image.save(pl.path.join(image_dir, image_basename), fake_images[i])
            image_basename = string.format('final_fake_images_%d.png', i, epoch)
            image.save(pl.path.join(image_dir, image_basename), fake_images[i])
        end
    end

    collectgarbage()
end

fake_losses = torch.Tensor(fake_losses)
info_losses_semsup = torch.Tensor(info_losses_semsup)
mean_semsup = string.format('%2.6f', info_losses_semsup:mean())
info_losses_unsup = torch.Tensor(info_losses_unsup)
mean_unsup = string.format('%2.6f', info_losses_unsup:mean())
real_losses = torch.Tensor(real_losses)
gen_losses = torch.Tensor(gen_losses)
x_val = torch.range(1, fake_losses:size(1))

gnuplot.pngfigure(outFolder .. "training.png")
gnuplot.plot(
    {'Fake losses', x_val, fake_losses, '~'},
    {'Info losses semsup - mean ' .. mean_semsup, x_val, info_losses_semsup, '~'},
    {'Info losses unsup - mean ' .. mean_unsup, x_val, info_losses_unsup, '~'},
    {'Real losses', x_val, real_losses, '~'},
    {'Gen losses', x_val, gen_losses, '~'}
)
gnuplot.plotflush()
gnuplot.close()
mean_class_losses_mean = string.format('%2.6f', mean_class_losses_1:mean())
x_val = torch.range(1, n_epochs)
gnuplot.pngfigure(outFolder .. "mean_class_losses_1.png")
gnuplot.plot(
    {'Mean class loss - mean ' .. mean_class_losses_mean, x_val, mean_class_losses_1, '~'}
)
gnuplot.axis{'','',0,1}
gnuplot.plotflush()
gnuplot.close()
infoHead_perf_mean_real = string.format('%2.6f', infoHead_performance[{{},{1}}]:squeeze():mean())
infoHead_perf_mean_synth_semsup =  string.format('%2.6f', infoHead_performance[{{},{2}}]:squeeze():mean())
infoHead_perf_mean_synth_unsup =  string.format('%2.6f', infoHead_performance[{{},{3}}]:squeeze():mean())

gnuplot.pngfigure(outFolder .. "infoHead_performance.png")
gnuplot.plot(
        {'Real data - mean ' .. infoHead_perf_mean_real, x_val, infoHead_performance[{{},{1}}]:squeeze(), '~'},
        {'Fake data semsup - mean ' .. infoHead_perf_mean_synth_semsup, x_val, infoHead_performance[{{},{2}}]:squeeze(), '~'},
        {'Fake data unsup - mean ' .. infoHead_perf_mean_synth_unsup, x_val, infoHead_performance[{{},{3}}]:squeeze(), '~'}
    )
gnuplot.axis{'','',0,1}
gnuplot.plotflush()
gnuplot.close()

gnuplot.pngfigure(outFolder .. "gradient_norms.png")
gnuplot.plot(
        {'Gradient norm D head', x_val, gradient_norms[{{},{1}}], '~'},
        {'Gradient norm Q ss', x_val, gradient_norms[{{},{2}}], '~'},
        {'Gradient norm Q us', x_val, gradient_norms[{{},{3}}], '~'}
    )
gnuplot.plotflush()
gnuplot.close()

metrics = {}
metrics.fake_losses = fake_losses
metrics.info_losses_semsup = info_losses_semsup
metrics.info_losses_unsup = info_losses_unsup
metrics.real_losses = real_losses
metrics.gen_losses = gen_losses
metrics.mean_class_losses_1 = mean_class_losses_1
metrics.infoHead_performance = infoHead_performance
torch.save(outFolder .. 'training_metrics.t7', metrics)
