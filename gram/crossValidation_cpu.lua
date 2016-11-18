require 'nn'

-- load data
file = torch.DiskFile('dat/facies_vectors.t7', 'r')
facies = file:readObject()
file:close()

names = {"shrimplin","alexander","shankle","luke","kimzey","cross","nolan","recruit","newby","churchman"}
perc_cor = {0,0,0,0,0,0,0,0,0,0}

for jj = 1,10 do
    
    name = names[jj]
    testing_data = {}
 
    -- rebuild the training_data table
    training_data = {}
    training_data["shrimplin"] = facies[{{1,471},{3,9}}]
    training_data["alexander"] = facies[{{472,937},{3,9}}]
    training_data["shankle"] = facies[{{938,1386},{3,9}}]
    training_data["luke"] = facies[{{1387,1847},{3,9}}]
    training_data["kimzey"] = facies[{{1848,2286},{3,9}}]
    training_data["cross"] = facies[{{2287,2787},{3,9}}]
    training_data["nolan"] = facies[{{2788,3202},{3,9}}]
    training_data["recruit"] = facies[{{3203,3282},{3,9}}]
    training_data["newby"] = facies[{{3283,3745},{3,9}}]
    training_data["churchman"] = facies[{{3746,4149},{3,9}}]
    -- rebuild the facies_label table
    facies_labels = {}
    facies_labels["shrimplin"] = facies[{{1,471},{1}}]
    facies_labels["alexander"] = facies[{{472,937},{1}}]
    facies_labels["shankle"] = facies[{{938,1386},{1}}]
    facies_labels["luke"] = facies[{{1387,1847},{1}}]
    facies_labels["kimzey"] = facies[{{1848,2286},{1}}]
    facies_labels["cross"] = facies[{{2287,2787},{1}}]
    facies_labels["nolan"] = facies[{{2788,3202},{1}}]
    facies_labels["recruit"] = facies[{{3203,3282},{1}}]
    facies_labels["newby"] = facies[{{3283,3745},{1}}]
    facies_labels["churchman"] = facies[{{3746,4149},{1}}]

    -- normalize the data
        -- training data
    mean = {}
    stdv = {}

    for key,value in pairs(training_data) do --over each well
        mean[key] = torch.Tensor(7)
        stdv[key] = torch.Tensor(7)
        for i = 1, 7 do --over each log
            mean[key][i] = training_data[key][{{},{i}}]:mean()
            training_data[key][{{},{i}}]:add(-mean[key][i])

            stdv[key][i] = training_data[key][{{},{i}}]:std()
            training_data[key][{{},{i}}]:div(stdv[key][i])
        end
    end
    
    -- chop out blind well
    blind_well = {}
    blind_labels = {}

    blind_well[name] = training_data[name][{{},{}}]
    training_data[name] = nil

    blind_labels[name] = facies_labels[name][{{},{}}]
    facies_labels[name] = nil
    
    -- condition the data
    trainset = {}

        -- the data
    trainset["data"] = torch.Tensor(facies:size()[1]-blind_well[name]:size()[1],7) 

    idx = 0
    for key,value in pairs(training_data) do
        for i = 1,training_data[key]:size()[1] do
            trainset["data"][i + idx] = training_data[key][i]
        end
        idx = idx + training_data[key]:size()[1]
    end

        -- the answers
    trainset["facies"] = torch.Tensor(facies:size()[1]-blind_labels[name]:size()[1])

    idx = 0
    for key,value in pairs(facies_labels) do
        for i = 1, facies_labels[key]:size()[1] do
            trainset["facies"][i + idx] = facies_labels[key][i]
        end
        idx = idx + facies_labels[key]:size()[1]
    end


    -- write index() and size() functions
    setmetatable(trainset, 
        {__index = function(t, i) 
                        return {t.data[i], t.facies[i]} 
                    end}
    );

    function trainset:size() 
        return self.data:size(1) 
    end

    -- condition the testing data
    testset = {}

        -- the data
    testset["data"] = torch.Tensor(blind_well[name]:size()[1],7) 

    for i = 1,blind_well[name]:size()[1] do
        testset["data"][i] = blind_well[name][i]
    end

        -- the answers
    testset["facies"] = torch.Tensor(blind_labels[name]:size()[1])

    for i = 1, blind_labels[name]:size()[1] do
        testset["facies"][i] = blind_labels[name][i]
    end

    setmetatable(testset, 
        {__index = function(t, i) 
                        return {t.data[i], t.facies[i]} 
                    end}
    );

    function testset:size() 
        return self.data:size(1) 
    end

    -- eliminate NaNs
    nan_mask = trainset.data:ne(trainset.data)
    trainset.data[nan_mask] = 0
    nan_mask = testset.data:ne(testset.data)
    testset.data[nan_mask] = 0

    -- build the neural net
    net = nil
    net = nn.Sequential()
    net:add(nn.Linear(7,200))
    net:add(nn.ReLU())
    net:add(nn.Linear(200,50))
    net:add(nn.ReLU())
    net:add(nn.Linear(50,9))
    net:add(nn.LogSoftMax())

    -- test the net -> forward
    input = torch.rand(7)
    output = net:forward(input)

    -- zero gradients and initialize
    net:zeroGradParameters()
    gradInput = net:backward(input, torch.rand(9))

    criterion = nn.ClassNLLCriterion()
    criterion:forward(output,3)
    gradients = criterion:backward(output, 3)

    gradInput = net:backward(input, gradients)

    -- train the net
    trainer = nn.StochasticGradient(net, criterion)
    trainer.learningRate = .001
    trainer.maxIteration = 20
    print("starting training")
    timer = torch.Timer()
    trainer:train(trainset)
    print("training time =", timer:time().real)
    
    -- overall performance
    correct = 0
    for i=1,testset:size() do
        local groundtruth = testset.facies[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end
    
    perc_cor[jj] = 100*correct/testset:size()
    
end

print("\n")
for i = 1,10 do
	print("well: ", names[i], "\tpercentage correct: ", perc_cor[i] .. " % \n")
end

