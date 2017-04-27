-------------------------------------------------
--
-- MODEL & CRITERION
--
-------------------------------------------------

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

disc_head_criterion = nn.BCECriterion()
info_head_semsup_crit = nn.CrossEntropyCriterion()
info_head_unsup_crit = nn.ParallelCriterion()

--- MODEL ---

generator = Seq()
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

discriminator_body = Seq()
    :add(Conv(1, 64, 4,4, 2,2, 1,1))
    :add(LeakyReLU())
    :add(Conv(64, 128, 4,4, 2,2, 1,1))
    :add(SpatBatchNorm(128))
    :add(LeakyReLU())
    :add(nn.Reshape(128 * 7 * 7))
    :add(Linear(128 * 7 * 7, 1024))
    :add(BatchNorm(1024))
    :add(LeakyReLU())

discriminator_head = Seq()
    :add(Linear(1024, 1))
    :add(nn.Sigmoid())

info_head_semsup = Seq()
    :add(Linear(1024, n_semsup_vars))

discriminator = Seq()
    :add(discriminator_body)

heads = nn.ConcatTable()
    :add(discriminator_head)
    :add(info_head_semsup)


info_head_unsup = nil
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

print('DISCRIMINATOR')
print(tostring(discriminator) .. '\n')
print('GENERATOR')
print(tostring(generator) .. '\n')
