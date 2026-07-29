{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "c3ab57c4-51a5-4b8f-be61-9190390eb31d",
   "metadata": {},
   "source": [
    "# GPU 버전"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "415db5d2-3b03-42c7-bbdc-897806c82207",
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install natsort"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f926c5c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "torch.cuda.is_available()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9c265986",
   "metadata": {},
   "outputs": [],
   "source": [
    "if torch.cuda.is_available():\n",
    "    print(torch.cuda.get_device_name(0))\n",
    "    print(torch.cuda.device_count())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f91274ef",
   "metadata": {},
   "source": [
    "# 데이터셋 선택 및 하이퍼파라미터 설정\n",
    "▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c52a3f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "import natsort\n",
    "import os\n",
    "\n",
    "#folder_list = os.listdir(\"./data/\")\n",
    "folder_list = ['cube']\n",
    "item_list = natsort.natsorted(folder_list)\n",
    "\n",
    "print(\"다음 데이터셋들이 학습됩니다 : \", item_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bedc091c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#최소10, 200~400 추천, 10단위로 pth가 저장됨, 여기선 50으로 진행\n",
    "epochs = 50\n",
    "batch_size = 16\n",
    "\n",
    "#3090 24GB에서 64까지 사용 가능했음\n",
    "learning_rate = 0.005\n",
    "image_size = 256"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3414afa4",
   "metadata": {},
   "source": [
    "▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "549ba101",
   "metadata": {},
   "source": [
    "# Main.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c20f67d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from dataset import get_data_transforms\n",
    "from torchvision.datasets import ImageFolder\n",
    "import numpy as np\n",
    "import random\n",
    "import os\n",
    "from torch.utils.data import DataLoader\n",
    "from resnet import resnet18, resnet34, resnet50, wide_resnet50_2\n",
    "from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50\n",
    "from dataset import RD_Dataset\n",
    "import torch.backends.cudnn as cudnn\n",
    "import argparse\n",
    "from torch.nn import functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9913144",
   "metadata": {},
   "outputs": [],
   "source": [
    "def setup_seed(seed):\n",
    "    torch.manual_seed(seed)\n",
    "    torch.cuda.manual_seed_all(seed)\n",
    "    np.random.seed(seed)\n",
    "    random.seed(seed)\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "    torch.backends.cudnn.benchmark = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a1f6b16",
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss_fucntion(a, b):\n",
    "    #mse_loss = torch.nn.MSELoss()\n",
    "    cos_loss = torch.nn.CosineSimilarity()\n",
    "    loss = 0\n",
    "    for item in range(len(a)):\n",
    "        #print(a[item].shape)\n",
    "        #print(b[item].shape)\n",
    "        #loss += 0.1*mse_loss(a[item], b[item])\n",
    "        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),\n",
    "                                      b[item].view(b[item].shape[0],-1)))\n",
    "    return loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1cf0108",
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss_concat(a, b):\n",
    "    mse_loss = torch.nn.MSELoss()\n",
    "    cos_loss = torch.nn.CosineSimilarity()\n",
    "    loss = 0\n",
    "    a_map = []\n",
    "    b_map = []\n",
    "    size = a[0].shape[-1]\n",
    "    for item in range(len(a)):\n",
    "        #loss += mse_loss(a[item], b[item])\n",
    "        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))\n",
    "        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))\n",
    "    a_map = torch.cat(a_map,1)\n",
    "    b_map = torch.cat(b_map,1)\n",
    "    loss += torch.mean(1-cos_loss(a_map,b_map))\n",
    "    return loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bb708b5c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(_class_):\n",
    "    print(_class_)\n",
    "        \n",
    "    device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "    print(device)\n",
    "\n",
    "    data_transform = get_data_transforms(image_size, image_size)\n",
    "    \n",
    "    train_path = './data/' + _class_ + '/train'\n",
    "    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'\n",
    "    os.makedirs('./checkpoints', exist_ok=True)\n",
    "    \n",
    "    train_data = ImageFolder(root=train_path, transform=data_transform)\n",
    "    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)\n",
    "\n",
    "    encoder, bn = wide_resnet50_2(pretrained=True)\n",
    "    encoder = encoder.to(device)\n",
    "    bn = bn.to(device)\n",
    "    encoder.eval()\n",
    "    decoder = de_wide_resnet50_2(pretrained=False)\n",
    "    decoder = decoder.to(device)\n",
    "\n",
    "    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))\n",
    "\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        start = time.time() \n",
    "        \n",
    "        bn.train()\n",
    "        decoder.train()\n",
    "        loss_list = []\n",
    "        for img, label in train_dataloader:\n",
    "            img = img.to(device)\n",
    "            inputs = encoder(img)\n",
    "            outputs = decoder(bn(inputs))#bn(inputs))\n",
    "            loss = loss_fucntion(inputs, outputs)\n",
    "            optimizer.zero_grad()\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            loss_list.append(loss.item())\n",
    "        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))\n",
    "        print(\"time :\",time.time() - start)  # 현재시각 - 시작시간 = 실행 시간\n",
    "        \n",
    "        if (epoch + 1) % 10 == 0:\n",
    "            torch.save({'bn': bn.state_dict(),'decoder': decoder.state_dict()}, ckp_path)\n",
    "            \n",
    "    return loss"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5fb3b48e",
   "metadata": {},
   "source": [
    "# 학습 시작"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5458778a",
   "metadata": {},
   "outputs": [],
   "source": [
    "setup_seed(111)\n",
    "\n",
    "import warnings\n",
    "warnings.simplefilter(action='ignore', category=FutureWarning)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2da1398e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#학습\n",
    "for i in item_list:\n",
    "    start_class = time.time()  # 시작 시간 저장\n",
    "\n",
    "    train(i)\n",
    "    print(i, \"time :\",time.time() - start_class)  # 현재시각 - 시작시간 = 실행 시간"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e2caf83",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.20"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
