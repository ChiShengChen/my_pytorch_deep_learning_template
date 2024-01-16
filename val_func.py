import torch
import tqdm
from cmal.utils import map_generate, highlight_im

def val(model, criterion, val_loader, device):
    model.eval()

    # Initialize the running loss and accuracy
    val_loss = 0.0
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects5 = 0
    
    # Iterate over the batches of the validation loader
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(val_loader):
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            # label_to_int = {label: idx for idx, label in enumerate(set(labels))}
            # numeric_labels = [label_to_int[label] for label in labels]
            # labels = torch.tensor(numeric_labels, dtype=torch.long).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, top3_pos = torch.topk(outputs.data, 5)

            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            batch_corrects4 = torch.sum((top3_pos[:, 3] == labels)).data.item()
            batch_corrects5 = torch.sum((top3_pos[:, 4] == labels)).data.item()
            val_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

            loss = criterion(outputs, labels)

            # Update the running loss and accuracy
            val_loss += loss.item() * inputs.size(0)

    # Calculate the validation loss and accuracy
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = float(val_corrects1) / len(val_loader.dataset)
    val5_acc = float(val_corrects5) / len(val_loader.dataset)
    return val_loss, val_acc, val5_acc

def val_prenet(model, criterion, val_loader, device):
    model.eval()
    val_loss = 0
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects5 = 0

    val_en_corrects1 = 0
    val_en_corrects2 = 0
    val_en_corrects5 = 0
    with torch.no_grad():
        for (inputs, targets) in tqdm.tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # label_to_int = {label: idx for idx, label in enumerate(set(targets))}
            # numeric_labels = [label_to_int[label] for label in targets]
            # targets = torch.tensor(numeric_labels, dtype=torch.long).to(device)

            _, _, _, output_concat, output1, output2, output3 = model(inputs, True)
            outputs_com = output1 + output2 + output3 + output_concat

            loss = criterion(output_concat, targets)
            val_loss += loss.item() * inputs.size(0)
            _, top3_pos = torch.topk(output_concat.data, 5)
            _, top3_pos_en = torch.topk(outputs_com.data, 5)


            batch_corrects1 = torch.sum((top3_pos[:, 0] == targets)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == targets)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos[:, 4] == targets)).data.item()
            val_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

            batch_corrects1 = torch.sum((top3_pos_en[:, 0] == targets)).data.item()
            val_en_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos_en[:, 1] == targets)).data.item()
            val_en_corrects2+= (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos_en[:, 2] == targets)).data.item()
            batch_corrects4 = torch.sum((top3_pos_en[:, 3] == targets)).data.item()
            batch_corrects5 = torch.sum((top3_pos_en[:, 4] == targets)).data.item()
            val_en_corrects5 += (batch_corrects5 + batch_corrects4 + batch_corrects3 + batch_corrects2 + batch_corrects1)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_corrects1 / len(val_loader.dataset)
    val5_acc = val_corrects5 / len(val_loader.dataset)
    val_acc_en = val_en_corrects1 / len(val_loader.dataset)
    val5_acc_en = val_en_corrects5 / len(val_loader.dataset)
    return val_loss, val_acc, val5_acc, val_acc_en, val5_acc_en 


def val_cmal(net, criterion, batch_size, testloader_in):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    correct_com2 = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    # testset = torchvision.datasets.ImageFolder(root=test_path,
    #                                            transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = testloader_in

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)
            output_1, output_2, output_3, output_concat, map1, map2, map3 = net(inputs)

            p1 = net.state_dict()['classifier3.1.weight']
            p2 = net.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output_3, p1, p2)

            p1 = net.state_dict()['classifier2.1.weight']
            p2 = net.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output_2, p1, p2)

            p1 = net.state_dict()['classifier1.1.weight']
            p2 = net.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output_1, p1, p2)

            inputs_ATT = highlight_im(inputs, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = net(inputs_ATT)

            outputs_com2 = output_1 + output_2 + output_3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

            loss = criterion(output_concat, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output_concat.data, 1)
            _, predicted_com = torch.max(outputs_com.data, 1)
            _, predicted_com2 = torch.max(outputs_com2.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_com += predicted_com.eq(targets.data).cpu().sum()
            correct_com2 += predicted_com2.eq(targets.data).cpu().sum()

            # if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f |Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1),
            100. * float(correct_com) / total, correct_com, total))

    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc_en, test_loss