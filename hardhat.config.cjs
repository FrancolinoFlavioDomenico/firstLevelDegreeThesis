require("@nomicfoundation/hardhat-toolbox");
/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: "0.8.28",
  networks:{
    hardhat:{
      accounts:{
        count:11,
        accountsBalance:"50000000000000000000000"
      },
      mining:{
        auto: false,
        interval: 5000,
        mempool:{
          order:"fifo"
        }
      },
      blockGasLimit:50000000,
      gas:"auto",
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
