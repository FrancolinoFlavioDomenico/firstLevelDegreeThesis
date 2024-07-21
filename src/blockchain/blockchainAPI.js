// import Fastify from 'fastify'
// import Web3 from 'web3';
// const web3 = new Web3(new Web3.providers.HttpProvider('http://127.0.0.2:8545'));
// const fastify = Fastify({
//   logger: true
// })

const fastify = require('fastify')({
  logger: true
})
const {Web3} = require('web3');
const web3 = new Web3(new Web3.providers.HttpProvider('http://127.0.0.2:8545'));


/*----------------------------------------------------------
        blockchain connection and tesContract test route
----------------------------------------------------------*/

//----------connection test-----------
fastify.get('/isConnected', async function (request, reply) {
  const isConnected =await web3.eth.net.isListening()
  reply.send({ isConnected: isConnected })
})

fastify.get('/web3/version', function (request, reply) {
  reply.send({ web3Version: `${Web3.version}` })
})

//---------testContract test---------------
//instantiate  testContract
const testContractAddress = '0x5b1869D9A4C187F2EAa108f3062412ecf0526b24'
const testContractAbi = require('./build/contracts/testContract.json').abi
const testContractIstance = new web3.eth.Contract(testContractAbi,testContractAddress)

//set contract string whit call (return error/null)
fastify.post('/contract/test/setTestStringCall', async function (request, reply) {
  let response = await testContractIstance.methods.setTestString('dinamic setted string').call({from: '0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1'})
  reply.send(response) 
})

//set contract string whit send (return succes)
fastify.post('/contract/test/setTestStringSend', async function (request, reply) {
  let response = await testContractIstance.methods.setTestString(request.body).send({from: '0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1'})
  reply.send(response.blockHash) 
})

//get string from  smart contract, setted from previous api
fastify.get('/contract/test/getTestString', async function (request, reply) {
  let response = await testContractIstance.methods.getTestString().call({from: '0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1'})
  reply.send(response) 
})

//get static string from  smart contract
fastify.get('/contract/test/getSimpleString', async function (request, reply) {
  let response = await testContractIstance.methods.getSimpleString().call({from: '0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1'})
  reply.send(response)
})





/*----------------------------------------------------------
                   real used api
----------------------------------------------------------*/
fastify.get('/getBlockchainAddress/:clientCid', async function (request, reply) {
  const response = await web3.eth.getAccounts();
  reply.send(response[request.params.clientCid])
})


















/*----------------------------------------------------------
              start server api
----------------------------------------------------------*/
// Run the server!
fastify.listen({ port: 3000 }, function (err, address) {
  if (err) {
    fastify.log.error(err)
    process.exit(1)
  }
  // Server is now listening on ${address}
})