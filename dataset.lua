-------------------------------------------------
--
-- DATASET
--
-------------------------------------------------

local tnt = require('torchnet')
local argcheck = require('argcheck')

local Dataset = torch.class('dataset', {})


Dataset.__init = argcheck{
  {name = 'self', type = 'dataset'},
  {name = 'data_file', type = 'string'},
  {name = 'unlabelled_prct', type = 'number'},
  {name = 'test_data_file', type = 'string', default = nil},
  {name = 'n_classes', type = 'number', default = 10},
  {name = 'batch_size', type = 'number', default = 128},
  {name = 'normalise', type = 'boolean', default = true},
  call = function(self, data_file, unlabelled_prct, test_data_file, n_classes, batch_size, normalise)
    local raw_data = torch.load(data_file, format)
    local dat = raw_data.data:float()
    -- Normalise data
    if normalise then
        dat = dat:float():div(255)
    end
    local ds_size = dat:size()
    local n_unlabelled = math.floor(ds_size[1] * unlabelled_prct)

    print('Number of unlabelled data: ' .. n_unlabelled .. ' of ' .. ds_size[1])

    local gen = torch.Generator()
    torch.manualSeed(gen, 1234)
    local idx = torch.randperm(gen, ds_size[1]):long()

    if unlabelled_prct > 0 then
        -- Get the unlabelled data
        self.inputs_unlabelled = dat:index(1, idx[{{1, n_unlabelled}}])
    end
    -- Get the labelled data
    if unlabelled_prct < 1 then
        self.inputs = dat:index(1, idx[{{n_unlabelled + 1, -1}}])
        self.targets = raw_data.labels:index(1, idx[{{n_unlabelled + 1, -1}}])
        -- Check number of instances of each class
        for i = 1, 10 do
            local nr_inst = self.targets:eq(i):sum()
            print('Number of instances of class ' .. i .. ': ' .. nr_inst)
        end
    end

    self.unlabelled_prct = unlabelled_prct

    if unlabelled_prct == 0 then
        print('\n' .. 'WARNING: No unlabelled data.' .. '\n' )
    end

    if unlabelled_prct == 1 then
        print('\n' ..'WARNING: No labelled data.' .. '\n' )
    end

    local test_inputs
    local lab = raw_data.labels
    if test_data_file ~= nil then
        t_dat = torch.load(test_data_file, format)
        lab = t_dat.labels
        dat = t_dat.data:float()
        if normalise then
            dat = dat:float():div(255)
        end
    end

    ds_size = dat:size()

    test_inputs = torch.CudaTensor(n_classes * batch_size, ds_size[2], ds_size[3], ds_size[4])
    for i=1, n_classes do
        -- Extract class i
        local mask = lab:eq(i)
        local idx = torch.range(1, ds_size[1])[mask]:long()
        local range1, range2
        range1 = (i-1) * batch_size + 1
        range2 = i * batch_size
        test_inputs[{{range1, range2}, {}}] = dat:index(1, idx[{{1, batch_size}}])
    end

    Dataset.test_inputs = test_inputs
  end
}

Dataset.make_iterator = argcheck{
  {name = 'self', type = 'dataset'},
  {name = 'batch_size', type = 'number', default = 32},
  {name = 'n_threads', type = 'number', default = 8},
  call = function(self, batch_size, n_threads)
    local inputs, targets = torch.Tensor{0}, torch.Tensor{0}

    if self.unlabelled_prct < 1 then
        inputs = self.inputs
          targets = self.targets
    end

    local function load_example_from_index(index)
      return {
        input = inputs[index],
        target = targets:narrow(1, index, 1)
      }
    end

    local gen = torch.Generator()
    torch.manualSeed(gen, 1234)
    local indices = torch.randperm(gen, inputs:size(1)):long()

    return tnt.ParallelDatasetIterator{
      ordered = true,
      nthread = n_threads,
      closure = function()
        local tnt = require('torchnet')

        return tnt.BatchDataset{
          batchsize = batch_size,
          dataset = tnt.ListDataset{
            list = indices,
            load = load_example_from_index
          }
        }
      end
    }
  end
}

Dataset.make_iterator_unlabelled = argcheck{
  {name = 'self', type = 'dataset'},
  {name = 'batch_size', type = 'number', default = 32},
  {name = 'n_threads', type = 'number', default = 8},
  call = function(self, batch_size, n_threads)
    local inputs, targets = torch.Tensor{0}, torch.Tensor{0}


    if self.unlabelled_prct > 0 then
        inputs = self.inputs_unlabelled
        targets = torch.zeros(inputs:size(1)):typeAs(targets):add(-1)
    end

    local function load_example_from_index(index)
      return {
        input = inputs[index],
        target = targets:narrow(1, index, 1)
      }
    end

    local gen = torch.Generator()
    torch.manualSeed(gen, 1234)
    local indices = torch.randperm(gen, inputs:size(1)):long()

    return tnt.ParallelDatasetIterator{
      ordered = true,
      nthread = n_threads,
      closure = function()
        local tnt = require('torchnet')

        return tnt.BatchDataset{
          batchsize = batch_size,
          dataset = tnt.ListDataset{
            list = indices,
            load = load_example_from_index
          }
        }
      end
    }
  end
}




return Dataset
