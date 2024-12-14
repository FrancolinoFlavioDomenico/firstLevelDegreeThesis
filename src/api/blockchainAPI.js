import Fastify from "fastify";
import { ethers } from "ethers";
import { createHeliaHTTP } from '@helia/http'
import { unixfs } from '@helia/unixfs'
import { createRequire } from "module";

const require = createRequire(import.meta.url);

const fastify = Fastify({
    logger: {
        enabled: true,
        file: "log/node_log.txt",
    },
    bodyLimit: 100 * 1024 * 1024,
});

fastify.register(require('@fastify/multipart'), {
    limits: {
        fileSize: 500 * 1024 * 1024,
    },
    attachFieldsToBody: "keyValues",
});

const provider = new ethers.JsonRpcProvider('http://127.0.0.1:8545/');
const helia = await createHeliaHTTP();
const fs = unixfs(helia)

const compiledContract = require("../blockchain/artifacts/src/blockchain/contracts/CheckWeights.sol/CheckWeights.json");
const contractABI = compiledContract.abi;
const contractBytecode = compiledContract.bytecode;
var deployedContractAddress;

/*********************************************************
 blockchain test route
 **********************************************************/

//----------connection test-----------
fastify.get("/isConnected", async function (request, reply) {
    const isConnected = await !!provider.getNetwork()
    reply.send({ isConnected: isConnected });
});


/*----------------------------------------------------------
                   real used api
----------------------------------------------------------*/

//Get blockchain address of selected client {:clientCid}//TODO REMOVE and set address by python array linked to client cid
fastify.get("/address/client/:clientCid", async function (request, reply) {
    const response = await provider.listAccounts();
    return response[request.params.clientCid].address;
});

//deploy smart contract on blockchain
fastify.post("/deploy/contract", async function (request, reply) {
    try {
        console.log(request.body.blockchainCredential)
        const wallet = new ethers.Wallet(request.body.blockchainCredential, provider)
        const contractFactory = new ethers.ContractFactory(contractABI, contractBytecode, wallet);
        const contract = await contractFactory.deploy();
        const deployTransaction = await contract.waitForDeployment();
        deployedContractAddress = deployTransaction.target
        reply.send({ deployTransaction: deployTransaction });
    } catch (e) {
        request.log.error({
            e,
            message: "deploy contract failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });
    } finally {
        return reply
    }
});

//write weights of client {:clientCid} at round {:round} into blockchain
fastify.post("/write/weights/:clientCid/:round", async (request, reply) => {

    try {
        const weights = request.body.weights
        let walletKey = request.body.blockchainCredential

        const wallet = new ethers.Wallet(walletKey, provider)
        const contract = new ethers.Contract(deployedContractAddress, contractABI, wallet);
        
        const weightAddress = await fs.addBytes(weights)

        const tx = await contract.storeWeightOfRoundForClient(
            weightAddress.toString(),
            request.params.round,
            request.params.clientCid,
        )
        const txResult = await tx.wait()
        reply.send({ txResult: txResult })
        return reply //yer or not?
    } catch (e) {
        request.log.error({
            e,
            message: "write file on blockchain failed:" + e,
        });
        reply.code(500).send({ error: "Internal Server Error" });

    } finally {
        return reply
    }
});


//write weights of client {:clientCid} at round {:round}
fastify.post("/weights/check/:round/:clientCid", async (request, reply) => {
    /* if weigth < avgWeitg of previous round
            add clientCid to blacklist
            write weigt on blockchain
            write blacklist on blockchain
      */
    try {
        const data = await request.file({
            limits: { fileSize: 500 * 1024 * 1024 },
        });
        const buffer = await data.toBuffer();
        console.log(JSON.parse(buffer.toString()));
        return reply.send();
    } catch (e) {
        request.log.error({
            e,
            message: "write weight on blockchain failed",
        });
        reply.log.error(e, `failed cid ${request.params.clientCid}`);
    } finally {
        return reply
    }
});

//Get blockchain blacklist
fastify.get("/blacklist", async function (request, reply) {
    return null;
});

/*----------------------------------------------------------
              start server
----------------------------------------------------------*/
// Run the server!
fastify.listen({ port: 3000 }, function (err, address) {
    if (err) {
        fastify.log.error(err);
        process.exit(1);
    }
    // Server is now listening on ${address}
});
