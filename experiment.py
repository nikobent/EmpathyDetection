import torch# If there's a GPU available...
import pandas as pd
from sklearn.model_selection import KFold
from transformers import BertTokenizer
from transformers import DistilBertTokenizer # second option to run locally
import helper
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import DistilBertForSequenceClassification #second option
from transformers import get_linear_schedule_with_warmup
import neptune
import time
import random
import numpy as np
import gc

# create the log file
# use gpu
# parameteres to examine
#1.Create loop that gets the hyperparameters for the model from a generator
    #clear memory
    #create the log section for these parameters
#2.Get the data and start the k-folds
    #tokenizer etc that do not need separation init
#3.Create the k-folds loop
    #a.save all results after each k
#4.Summurized k-folds results on these settings
    #save on neptune at the end


#create the log file
log_file = open("log_kfold_empathy.txt","a")
#change 1
log_file.write("This is the log file for DistillBERT CV using 5-fold crossvalidation.\n")
log_file.write("For each parameter setting, we are performing 5-fold crossvalidation and save the loss and pearsonR for "
               "each iteration and the average at the end.\n")

#use gpu
#Check if gpu available
if torch.cuda.is_available():        # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    log_file.write('\nThere are %d GPU(s) available.' % torch.cuda.device_count())
    log_file.write('\nWe will use the GPU:'+ str(torch.cuda.get_device_name(0)))# If not...
else:
    log_file.write('\nNo GPU available, using the CPU instead.')
    device = torch.device("cpu")

#parameteres and generator
lrs= [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
weight_decays = [0, 0.1, 0.2, 0.3, 0.4]
batches = [8,16]
myParGen = helper.param_gener(lrs,weight_decays,batches)

#read data
data = pd.read_csv("messages_90.csv")
x = data.essay
y = data.empathy

counter = 0
while(True):
    #clear from previous model
    torch.cuda.empty_cache()
    #counter for experiment
    counter = counter + 1
    lr, weight_decay, per_gpu_batch_size = next(myParGen)
    #if no more parameters are returned
    if (type(lr) != float):
        log_file.write("\n experiment is over")
        log_file.close()
        break
    log_file.write("\n!!!!!!!!!!!!!!!!!!!Model number "+str(counter)+"!!!!!!!!!!!!!!!!!\n")
    print("\n!!!!!!!!!!!!!!!!!!!Model number "+str(counter)+"!!!!!!!!!!!!!!!!!\n")
    log_file.write("Parameteres: lr "+ str(lr)+ ", weight_decay "+str(weight_decay)+ ", batch "+ str(per_gpu_batch_size)+".\n")

    kf = KFold(n_splits=5)
    split = kf.split(data)
    losses_train = 0
    avg_k_loss_train = 0
    correlations_test = 0
    losses_test = 0
    avg_k_corr_test = 0
    avg_k_loss_test = 0
    #counter for k in k-fold
    counter2 = 0
    for train_index, test_index in split:
        torch.cuda.empty_cache()
        counter2 = counter2 +1
        log_file.write(str(counter2)+" fold starting now\n")

        #change 2 for BERT or distilled or roberta
        # Load the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  # not necessary to lower, the data are already transformed

        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = helper.tokenize(tokenizer, x)

        # after tokenizing we must perform the padding
        # the max sentence length is 201
        MAX_LEN = 256
        # Pad our input tokens with value 0.
        # "post" indicates that we want to pad and truncate at the end of the sequence,
        # as opposed to the beginning.
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                                  value=0, truncating="post", padding="post")

        # after padding then comes the attention masks
        attention_masks = helper.masks(input_ids)

        # time to split CHOOSE EMPATHY OR DISTRESS HERE
        # Use 90% for training and 10% for validation.
        train_inputs = input_ids[train_index]
        validation_inputs = input_ids[test_index]
        train_labels = y[train_index]
        validation_labels = y[test_index]
        train_masks = np.array(attention_masks)[train_index]
        validation_masks = np.array(attention_masks)[test_index]
        # Convert all inputs and labels into torch tensors, the required datatype
        # for our model.
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels.values)
        validation_labels = torch.tensor(validation_labels.values)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        # The DataLoader needs to know our batch size for training, so we specify it
        # here.
        # For fine-tuning BERT on a specific task, the authors recommend a batch size of
        # 16 or 32.

        # here dataloader etc
        batch_size = per_gpu_batch_size
        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        # Load BertForSequenceClassification, the pretrained BERT model with a single
        # linear classification layer on top.
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=1,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        model.cuda()

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
                          weight_decay=weight_decay
                          )

        # Number of training epochs. The BERT authors recommend between 2 and 4.
        # We chose to run for 4, but we'll see later that this may be over-fitting the
        # training data.
        epochs = 3

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # !!!!!!start training phase!!!
        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            log_file.write("\n")
            log_file.write('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            log_file.write('\nTraining...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = helper.format_time(time.time() - t0)

                    # Report progress.
                    log_file.write('\n Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_labels = b_labels.type(torch.cuda.FloatTensor)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                # b_labels = b_labels.long()
                b_input_ids = b_input_ids.type(torch.int64)
                #b_input_ids = torch.tensor(b_input_ids).to(torch.int64)
                loss = model(b_input_ids,
                             attention_mask=b_input_mask,
                             labels=b_labels)[0]
                #token_type_ids=None,
                loss = loss.type(torch.cuda.FloatTensor)
                logits = model(b_input_ids,
                               attention_mask=b_input_mask,
                               labels=b_labels)[1]
                #token_type_ids = None,
                logits = logits.type(torch.cuda.FloatTensor)
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += float(loss.item())

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)


            # Measure how long this epoch took.
            tt = time.time() - t0
            training_time = helper.format_time(time.time() - t0)

            log_file.write("\n")
            log_file.write("\n  Average training loss: {0:.2f}".format(avg_train_loss))
            log_file.write("\n  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            log_file.write("\n")
            log_file.write("\nRunning Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            total_eval_cor = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                b_labels = b_labels.type(torch.cuda.FloatTensor)


                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    b_input_ids = b_input_ids.type(torch.int64)
                    loss = model(b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)[0]
                    # token_type_ids=None,
                    loss = loss.type(torch.cuda.FloatTensor)
                    logits = model(b_input_ids,
                                   attention_mask=b_input_mask,
                                   labels=b_labels)[1]
                    # token_type_ids=None,
                    logits = logits.type(torch.cuda.FloatTensor)

                # Accumulate the validation loss.
                total_eval_loss += float(loss.item())  # put the float to avoid cuda out of memory


                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_cor += helper.corr(logits, label_ids)[0]

            # Report the final accuracy for this validation run.
            avg_val_cor = total_eval_cor / len(validation_dataloader)
            log_file.write("\n  cor: {0:.2f}".format(avg_val_cor))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = helper.format_time(time.time() - t0)

            log_file.write("\n  Validation Loss: {0:.2f}".format(avg_val_loss))
            log_file.write("\n  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. cor.': avg_val_cor,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        losses_test = losses_test + avg_val_loss
        correlations_test = correlations_test + avg_val_cor
        losses_train = losses_train + avg_train_loss
        log_file.write("\n")
        log_file.write("\nTraining complete!")

        log_file.write("\nTotal training took {:} (h:mm:ss)".format(helper.format_time(time.time() - total_t0)))

        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)
    avg_k_loss_test = losses_test/5
    avg_k_loss_train = losses_train/5
    avg_k_corr_test = correlations_test/5

    log_file.write("The average results for this models after completing all the folds:\n")
    log_file.write("average training loss on final epoch :"+ str(avg_k_loss_train)+"\n")
    log_file.write("average validation loss on final epoch :"+ str(avg_k_loss_test)+"\n")
    log_file.write("average validation correlation on final epoch :"+str(avg_k_corr_test)+"\n")

    #for keeping stats
    neptune.init(project_qualified_name='nikobent/BERTReactions5fold',
                 api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOGEzYmVkMTQtMDEzZC00NmYyLThjM2EtZWMzNTc4YTgzMWRhIn0=',
                 )
    params = {
      "per_gpu_batch_size": per_gpu_batch_size,
      "weight_decay": weight_decay,
      "learning_rate":  lr,
      "num_epochs": 3
    }
    neptune.create_experiment(name="BertBase", params=params)
    neptune.log_metric('Training Loss 5 fold', avg_k_loss_train)
    neptune.log_metric('Valid. Loss 5 fold ', avg_k_loss_test)
    neptune.log_metric('Valid. cor. 5 fold ', avg_k_corr_test)
    del train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks
    del tokenizer, logits, loss, b_input_ids, b_input_mask, b_labels, model, input_ids
    del train_data, train_sampler, train_dataloader, validation_data, validation_sampler, validation_dataloader, scheduler
    print(torch.cuda.memory_summary())
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            del obj
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary())
