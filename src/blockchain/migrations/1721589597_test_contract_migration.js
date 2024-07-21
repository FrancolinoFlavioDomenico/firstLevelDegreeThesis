var testContract = artifacts.require("testContract");

module.exports = function(deployer) {
  // deployment steps
  deployer.deploy(testContract);
};
