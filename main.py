# ==============================
# 1. Setup
# ==============================
!pip install torch torchvision matplotlib numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

%matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==============================
# 2. Prunable Layer
# ==============================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


# ==============================
# 3. Model
# ==============================
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ==============================
# 4. Loss Functions
# ==============================
def compute_sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss


# ==============================
# 5. Dataset (Normalized ✔)
# ==============================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)


# ==============================
# 6. Training
# ==============================
def train(model, lambda_sparse=1e-4, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            cls_loss = criterion(outputs, labels)
            sp_loss = compute_sparsity_loss(model)

            loss = cls_loss + lambda_sparse * sp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")


# ==============================
# 7. Evaluation
# ==============================
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# ==============================
# 8. Sparsity Calculation
# ==============================
def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)

            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total


# ==============================
# 9. Plot Gates (FIXED ✔)
# ==============================
def plot_gates(model, title="Gate Distribution"):
    all_gates = []

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    print("Total gates:", len(all_gates))

    plt.figure()
    plt.hist(all_gates, bins=50)
    plt.title(title)
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
    plt.show()


# ==============================
# 10. Run Experiments
# ==============================
lambdas = [1e-5, 1e-4, 1e-3]

results = []
models = []

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")

    model = PrunableNet().to(device)

    train(model, lambda_sparse=lam, epochs=5)

    acc = evaluate(model)
    sparsity = compute_sparsity(model)

    results.append((lam, acc, sparsity))
    models.append(model)

    print(f"Accuracy: {acc:.2f}%, Sparsity: {sparsity:.2f}%")


# ==============================
# 11. Results Table
# ==============================
print("\nFinal Results:")
for r in results:
    print(f"Lambda: {r[0]}, Accuracy: {r[1]:.2f}, Sparsity: {r[2]:.2f}%")


# ==============================
# 12. Plot Distributions
# ==============================
for i, lam in enumerate(lambdas):
    plot_gates(models[i], title=f"Lambda = {lam}")
