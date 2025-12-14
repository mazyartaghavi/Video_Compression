class PPOAgent:
    def __init__(self, policy, lr=3e-5):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
