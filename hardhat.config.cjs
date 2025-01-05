require("@nomicfoundation/hardhat-toolbox");
/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.28",
  networks:{
    hardhat:{
      accounts:{
        count:11,
        accountsBalance:"100000000000000000000000"
      },
      mining:{
        auto: false,
        interval: 5000,
        mempool:{
          order:"fifo"
        }
      },
      blockGasLimit:10000000,
      gas:10000000,
      gasPrice:1,
      initialBaseFeePerGas: "100"
    }
  },
  paths:{
    artifacts: "src/blockchain/artifacts",
    tests: "src/blockchain/test",
    sources: "src/blockchain/contracts",
    ignition: "src/blockchain/ignition",
    cache: "data/tmp"
  },
};
