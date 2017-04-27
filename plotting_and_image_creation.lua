-------------------------------------------------
--
-- PLOTTING & IMAGE CREATION
--
-------------------------------------------------

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

-- Print input for G to create synthetic samples
local n_classes = n_classes_per_lc[1]
local G_print_input = {}
local nr_inputs_G_print_per_class = 12
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

function create_images()
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

    return fake_images
end

function plot_final_metrics()
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
    if classifier_set then
        mean_class_losses_mean = string.format('%2.6f', mean_class_losses_1:mean())
        x_val = torch.range(1, n_epochs)
        gnuplot.pngfigure(outFolder .. "mean_class_losses_1.png")
        gnuplot.plot(
            {'Mean class loss - mean ' .. mean_class_losses_mean, x_val, mean_class_losses_1, '~'}
        )
        gnuplot.axis{'','',0,1}
        gnuplot.plotflush()
        gnuplot.close()
    end
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

    -- Save the metrics for potential future analysis
    metrics = {}
    metrics.fake_losses = fake_losses
    metrics.info_losses_semsup = info_losses_semsup
    metrics.info_losses_unsup = info_losses_unsup
    metrics.real_losses = real_losses
    metrics.gen_losses = gen_losses
    metrics.mean_class_losses_1 = mean_class_losses_1
    metrics.infoHead_performance = infoHead_performance
    torch.save(outFolder .. 'training_metrics.t7', metrics)
end
