-------------------------------------------------
--
-- CONFIG FILE
--
-------------------------------------------------


-- Training parameters
n_epochs = 50
n_updates_per_epoch = 100
batch_size = 128
disc_learning_rate = 1e-4
gen_learning_rate = 1e-3
rng_seed = 1

-- The coefficient with which the discriminator should train Q_us and Q_ss
info_reg_sup_coef_train = 2
info_reg_unsup_coef_train = 0.8

-- The coefficient with which the generator should learn with Q_us and Q_ss
info_reg_sup_coef_learn = 2
info_reg_unsup_coef_learn = 0.8

n_classes_per_lc = {1, 10}
c_offset = torch.totable(torch.cumsum(torch.Tensor(n_classes_per_lc)))
table.remove(n_classes_per_lc, 1)
-- At this point, n_classes_per_lc[i] is the number of classes the i-th
-- categorical latent code contains.
-- c_offset[i] corresponds to the starting index of categorical code i.

-- Helper variables
n_cat_vars = c_offset[#c_offset] - 1
n_cont_vars = 2
n_noise_vars = 62
n_semsup_vars = n_classes_per_lc[1]
n_unsup_vars = n_cat_vars - n_semsup_vars + n_cont_vars
n_gen_inputs = n_cont_vars + n_cat_vars + n_noise_vars
n_salient_vars = n_cont_vars + n_cat_vars

-- Percentage of data to unlabel
unlabelled_percentage = 0.99
-- f true, sets the unlabeled sampling probability to 0 and grows it linearly
-- to the true value. Useful if the number of labeled data is small.
sampling_prob_growth = true
additive_prob_growth = 0.1
unlabelled_sampling_prob = unlabelled_percentage
-- If enabled, it will grow the unlabelled sampling probability in each epoch until the value of
-- unlabelled_percentage
if sampling_prob_growth and unlabelled_percentage < 1 then
    unlabelled_sampling_prob = 0.0
end

-- Instance noise parameters
instance_noise_std = 0
instance_noise_annealing = 0
classifier_path = ''
classifier_set = true
if not classifier_path or classifier_path == '' then
    print('WARNING: No classifier specified. \n')
    classifier_set = false
end
dataset_train_path = ''
dataset_test_path = ''
samples_dim = {1, 28, 28}

assert(not (dataset_train_path == '' or dataset_test_path == ''), 'One of the data paths is not set.')

local function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- Set manual seeds for reproducible results
torch.manualSeed(rng_seed)
cutorch.manualSeedAll(rng_seed)
math.randomseed(rng_seed)

torch.setdefaulttensortype('torch.FloatTensor')

-- Create a non-existant experiment folder
outFolder = 'exp1/'
i = 1
while file_exists(outFolder) do
  i = i + 1
  outFolder = 'exp' .. i .. '/'
end
print('Output directory: ' .. outFolder .. '\n')
lfs.mkdir(outFolder)
