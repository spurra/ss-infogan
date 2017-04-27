--[[
    This code is a heavily modified version of and based on the code found in

    https://github.com/anibali/infogan
]]--

-------------------------------------------------
--
-- MAIN CODE & TRAINING
--
-------------------------------------------------

-- STANDARD PACKAGES
require('torch')
require('image')
require('nn')
require('optim')

-- NON-STANDARD PACKAGES
-- Bindings to CUDA
require('cutorch')
require('cunn')
require('cudnn')
-- Plotting library
require 'gnuplot'
pl = require('pl.import_into')()
tnt = require('torchnet')
nninit = require('nninit')

-- CUSTOM SCRIPTS
-- Contains all parameters and paths
require 'configuration'
-- Contains helper functions for training and evaluation
require 'helper_functions'
-- Creates the model and criterion functions
require 'model_and_criterion'
-- Contains functions for plotting and synthetic image creation
require 'plotting_and_image_creation'
-- Creates a dataset object to return labeled and unlabeled training batches
dataset = require('dataset')

--[[
    INITIATLISATION
]]

if classifier_set then
    C1 = torch.load(classifier_path):cuda()
end
train_data = dataset.new(dataset_train_path, unlabelled_percentage, dataset_test_path)
train_iter_labelled = train_data:make_iterator(batch_size)
train_iter_unlabelled = train_data:make_iterator_unlabelled(batch_size)

-- Save the current parameters
params = {}
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
params.G = tostring(generator)
params.D = tostring(discriminator)
torch.save(outFolder .. 'params.t7', params)


--[[
    LOGGING
]]

log_text = require('torchnet.log.view.text')

log_keys = {'epoch', 'fake_loss', 'info_loss_semsup','info_loss_unsup',
  'real_loss', 'gen_loss', 'time'}

log = tnt.Log{
  keys = log_keys,
  onFlush = {
    log_text{
      keys = log_keys,
      format = {'epoch=%3d', 'fake_loss=%8.5f', 'info_loss_semsup=%8.6f', 'info_loss_unsup=%8.6f',
        'real_loss=%8.6f', 'gen_loss=%8.6f', 'time=%5.2fs'}
    }
  }
}


--[[
    TRAINING
]]

real_input = torch.CudaTensor()
gen_input = torch.CudaTensor()
fake_input = torch.CudaTensor()
disc_target = torch.CudaTensor()
target = torch.CudaTensor()

disc_params, disc_grad_params = discriminator:getParameters()
gen_params, gen_grad_params = generator:getParameters()

fake_loss_meter = tnt.AverageValueMeter()
info_loss_semsup_meter = tnt.AverageValueMeter()
info_loss_unsup_meter = tnt.AverageValueMeter()
real_loss_meter = tnt.AverageValueMeter()
gen_loss_meter = tnt.AverageValueMeter()
time_meter = tnt.TimeMeter()

mean_class_losses_1 = torch.Tensor(n_epochs)
infoHead_performance = torch.Tensor(n_epochs, 3)
gradient_norms = torch.Tensor(n_epochs, 3)

-- Calculate outputs and gradients for the discriminator
do_discriminator_step = function(new_params)
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
    info_loss_semsup_meter:add(loss_info_semsup)
    info_loss_unsup_meter:add(loss_info_unsup)

    -- Calculate combined loss
    local loss = loss_real + loss_fake + loss_info

    -- Anneal instance noise
    instance_noise_std = instance_noise_std * instance_noise_annealing

    return loss, disc_grad_params
end

-- Calculate outputs and gradients for the generator
do_generator_step = function(new_params)
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
disc_optimiser = {
        method = optim.adam,
        config = {
        epsilon = 1e-8,
        learningRate = disc_learning_rate,
        beta1 = 0.5
    },
    state = {}
}

-- Generator optimiser
gen_optimiser = {
        method = optim.adam,
        config = {
        epsilon = 1e-8,
        learningRate = gen_learning_rate,
        beta1 = 0.5
    },
    state = {}
}


iter_inst_labelled = train_iter_labelled()
iter_inst_unlabelled = train_iter_unlabelled()

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

    discriminator:training()
    generator:training()

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

    fake_images = create_images()

  -- Update log
    log:set{
        epoch = epoch,
        fake_loss = fake_loss_meter:value(),
        info_loss_semsup = info_loss_semsup_meter:value(),
        info_loss_unsup = info_loss_unsup_meter:value(),
        real_loss = real_loss_meter:value(),
        gen_loss = gen_loss_meter:value(),
        time = time_meter:value()
    }
    log:flush()

    -- Save metrics
    info_losses_semsup[#info_losses_semsup+1] = info_loss_semsup_meter:value()
    info_losses_unsup[#info_losses_unsup+1] = info_loss_unsup_meter:value()
    real_losses[#real_losses+1] = real_loss_meter:value()
    gen_losses[#gen_losses+1] = gen_loss_meter:value()
    fake_losses[#fake_losses+1] = fake_loss_meter:value()
    infoHead_performance[{{epoch}, {1}}] = evaluate_infoHead_real(discriminator)
    infoHead_performance[{{epoch}, {2}}], infoHead_performance[{{epoch}, {3}}] = evaluate_infoHead_fake(generator, discriminator)
    if classifier_set then
        mean_class_losses_1[epoch] = test_signal(generator, C1)
    end

    -- Save the discriminator and generator models
    local model_dir = pl.path.join(outFolder, 'models')
    pl.dir.makepath(model_dir)
    discriminator:clearState()
    local model_disc_file = pl.path.join(model_dir, 'infogan_svhn_disc.t7')
    torch.save(model_disc_file, discriminator)
    generator:clearState()
    local model_gen_file = pl.path.join(model_dir, 'infogan_svhn_gen.t7')
    torch.save(model_gen_file, generator)

    -- Save images every 10 epochs
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

plot_final_metrics()
