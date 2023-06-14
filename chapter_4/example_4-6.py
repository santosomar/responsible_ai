# Performing the membership inference attack
attack_model = Net().to(device)
optimizer = optim.SGD(attack_model.parameters(), lr=0.001, momentum=0.9)

# Train the attack model on the outputs of the shadow model
for epoch in range(10):  # loop over the dataset multiple times
    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        shadow_outputs = shadow_model(inputs)
        attack_outputs = attack_model(shadow_outputs.detach())
        loss = criterion(attack_outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training the Attack Model')

# Check if the samples from the test_loader were in the training set of the target model
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = attack_model(target_model(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the attack model: %d %%' % (100 * correct / total))
