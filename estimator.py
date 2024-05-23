import torch


class ZOGradientEstimator:
    '''
        ZOEstimator: estimate gradient(mini-batch or full-batch) on every period
        save the previous checkpoint of parameters and full-batch gradient
    '''

    def __init__(self, model, device, dataloader, seed=0):
        self.batch = None
        self.model = model
        self.device = device
        self.data_loader = dataloader
        self.current_iter = iter(self.dataloader)
        self.saved_params = {}
        self.copy_params = {}
        torch.manual_seed(seed)

    def compute_zeroth_order_grad(self, batch, current=True, batch_mode=True, epsilon=1e-4):  # current表示是否计算当前时间点的梯度
        # Set the random seed for reproducibility
        self.batch = batch
        grads = [torch.zeros_like(param) for param in self.model.parameters()]

        if current:
            if batch_mode:
                # estimate on the current batch
                # Iterate over the data loader for mini-batch gradient estimation
                '''
                data_iter = self.current_iter
                try:
                    batch = next(data_iter)  # 获取下一个小批量
                except StopIteration:
                    # 如果到达数据集末尾，重置迭代器并再次获取第一个小批量
                    self.data_iter = iter(self.data_loader)
                    batch = next(self.data_iter)
                '''
                grads = self.compute_batch_grad(self.batch, epsilon)
            else:
                # Compute full-batch gradient estimation
                grads = self.compute_full_batch_grad(self.data_loader, epsilon)
        else:
            '''
            data_iter = self.current_iter
            try:
                batch = next(data_iter)  # 获取下一个小批量
            except StopIteration:
                # 如果到达数据集末尾，重置迭代器并再次获取第一个小批量
                self.data_iter = iter(self.data_loader)
                batch = next(self.data_iter)
            '''
            grads = self.compute_pre_batch_grad(self.batch, epsilon)

        return grads

    def compute_batch_grad(self, batch, epsilon):
        grads = []
        for param in self.model.parameters():
            perturb = torch.randn_like(param) * epsilon

            # Compute perturbed loss (positive direction)
            param.data.add_(perturb)
            loss = self.compute_loss(batch).item()
            loss_perturbed1 = loss

            # Compute perturbed loss (negative direction)
            param.data.sub_(2 * perturb)
            loss = self.compute_loss(batch).item()
            loss_perturbed2 = loss

            # Compute the zeroth order gradient estimate for the current parameter
            grad = ((loss_perturbed1 - loss_perturbed2) / (2 * epsilon)) * torch.randn_like(param)

            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.data.copy_(grad)

            grads.append(grad)
            param.data.add_(perturb)

        return grads

    def compute_full_batch_grad(self, epsilon):
        grads = []
        for param in self.model.parameters():
            perturb = torch.randn_like(param) * epsilon

            # Compute perturbed loss (positive direction)
            param.data.add_(perturb)

            total_loss = 0
            for i, batch in enumerate(self.data_loader):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
            avg_loss = total_loss / len(self.data_loader)
            loss_perturbed1 = avg_loss

            # Compute perturbed loss (negative direction)
            param.data.sub_(2 * perturb)
            total_loss = 0
            for i, batch in enumerate(self.data_loader):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
            avg_loss = total_loss / len(self.data_loader)
            loss_perturbed2 = avg_loss

            # Compute the zeroth order gradient estimate for the current parameter
            grad = ((loss_perturbed1 - loss_perturbed2) / (2 * epsilon)) * torch.randn_like(param)

            if param.grad is None:
                param.grad = torch.zeros_like(param)
            else:
                param.grad.data.copy_(grad)

            grads.append(grad)
            # recover the original parameters
            param.data.add_(perturb)

        self.saved_params()
        return grads

    def compute_pre_batch_grad(self, batch, epsilon):
        grads = []
        for param in self.model.parameters():
            perturb = torch.randn_like(param) * epsilon

            # Compute perturbed loss (positive direction)
            param.data.add_(perturb)
            loss = self.compute_loss(batch).item()
            loss_perturbed1 = loss

            # Compute perturbed loss (negative direction)
            param.data.sub_(2 * perturb)
            loss = self.compute_loss(batch).item()
            loss_perturbed2 = loss

            # Compute the zeroth order gradient estimate for the current parameter
            grad = ((loss_perturbed1 - loss_perturbed2) / (2 * epsilon)) * torch.randn_like(param)

            if param.grad is None:
                param.pre_grad = torch.zeros_like(param)
            else:
                param.pre_grad.data.copy_(grad)

            grads.append(grad)
            param.data.add_(perturb)

        return grads

    def compute_loss(self, batch):
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.tokenizer.model_input_names}
        labels = batch['label'].to(self.device)
        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    def save_params(self):
        for n, p in self.model.named_parameters():
            self.saved_params[n].copy_(p.data)

    def copy_params(self):
        for n, p in self.model.named_parameters():
            self.copy_params[n].copy_(p.data)

    def restore(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self.saved_params[n])

    def reload_params(self):
        for n, p in self.model.named_parameters():
            p.data.copy_(self.copy_params[n])
